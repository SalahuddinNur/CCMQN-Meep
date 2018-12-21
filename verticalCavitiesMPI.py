import os
import datetime
import numpy as np
import meep as mp

# refractive indices
n_SiN = 2.01
n_SiO2 = 1.46
n_TiO2 = 2.35
n_ITO = 1.85

n_high = n_TiO2
n_low = n_SiO2
n_electrode = n_ITO

# Defect resolution
RES = 1/10000


def system_command(command):
    if mp.am_master():
        os.system(command)


class DbrMirror:

    def __init__(self, n_ac, dbr_periodicity, dbr_fill_factor, dbr_number):
        now = datetime.datetime.now()
        self.directory = 'log/' + now.strftime("%Y-%m-%d") + '/'

        self.n_ac = n_ac
        self.dbr_periodicity = dbr_periodicity
        self.dbr_fill_factor = dbr_fill_factor
        self.dbr_number = dbr_number

        self._cell = None
        self._geometry = None
        self._sources = None
        self._simulation = None


    def define_cell(self, pad_z=1, dpml=1):
        self.pad_z = pad_z
        self.dpml = dpml

        self.sz = self.dbr_periodicity * self.dbr_number + 2 * (self.pad_z + self.dpml)
        self._cell = mp.Vector3(self.sz, 0, 0)

    def define_geometry(self, norm=False):
        if self._cell is not None:
            # define geometry
            self._geometry = []

            if not norm:
                # from bottom
                current_layer_position = - self.sz / 2 + self.pad_z + self.dpml

                # Oxide / Nitride pairs
                for i in range(self.dbr_number):
                    current_layer_thickness = self.dbr_periodicity * (1 - self.dbr_fill_factor)
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(current_layer_position + current_layer_thickness / 2),
                                 size=mp.Vector3(current_layer_thickness, 1e20, 1e20),
                                 material=mp.Medium(index=n_low)))
                    current_layer_position += current_layer_thickness

                    current_layer_thickness = self.dbr_periodicity * self.dbr_fill_factor
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(current_layer_position + current_layer_thickness / 2),
                                 size=mp.Vector3(current_layer_thickness, 1e20, 1e20),
                                 material=mp.Medium(index=n_high)))
                    current_layer_position += current_layer_thickness

                # final Oxide layer
                self._geometry.append(
                    mp.Block(center=mp.Vector3(current_layer_position + (self.pad_z + self.dpml) / 2),
                             size=mp.Vector3(self.pad_z + self.dpml, 1e20, 1e20),
                             material=mp.Medium(index=n_low)))

        else:
            raise Exception('Need a cell before defining geometry')

    def define_sources(self, fcen=1.27389, df=0.1):
        if self._geometry is not None:
            self.fcen = fcen
            self.df = df
            self._sources = [mp.Source(src=mp.GaussianSource(fcen, fwidth=df),
                                       component=mp.Ez,
                                       center=mp.Vector3(- self.sz / 2 + self.pad_z / 3 + self.dpml))]
        else:
            raise Exception('Need a geometry before defining sources')

    def define_simulation(self, res=32):
        if (self._cell is not None) & (self._geometry is not None) & (self._sources is not None):
            self.res = res
            pml_layers = [mp.PML(self.dpml)]
            self._simulation = mp.Simulation(cell_size=self._cell,
                                             geometry=self._geometry,
                                             boundary_layers=pml_layers,
                                             resolution=res,
                                             sources=self._sources,
                                             default_material=mp.Medium(index=self.n_ac))
        else:
            raise Exception('Environment not fully defined. Need a cell, geometry and sources')

    def define_flux_regions(self, nfreq=100):
        if self._simulation is not None:
            self.nfreq = nfreq
            refl_fr = mp.FluxRegion(center=mp.Vector3(- self.sz / 2 + 2 * self.pad_z / 3 + self.dpml))
            self._refl_flux = self._simulation.add_flux(self.fcen, self.df, nfreq, refl_fr)
            trans_fr = mp.FluxRegion(center=mp.Vector3(+ self.sz / 2 - 2 * self.pad_z / 3 - self.dpml))
            self._trans_flux = self._simulation.add_flux(self.fcen, self.df, nfreq, trans_fr)

        else:
            raise Exception('Cannot add flux region to null simulation')

    def end_simulation(self):
        self._simulation.reset_meep()
        self._cell = None
        self._geometry = None
        self._sources = None
        self._simulation = None

    def get_epsilon(self, res=32):
        self.define_cell()
        self.define_geometry(norm=False)
        self.define_sources()
        self.define_simulation(res=res)
        self._simulation.run(until=1 / self.fcen)
        eps_data = self._simulation.get_array(mp.Vector3(), mp.Vector3(self.sz),
                                              mp.Dielectric)[::-1]
        return eps_data

    def get_spectra(self, fcen=1.27389, df=1.0, res=32, nfreq=100, time_after_source=200):

        self.time_after_sources = time_after_source
        self.define_cell()
        self.define_geometry(norm=True)
        self.define_sources(fcen=fcen, df=df)
        self.define_simulation(res=res)
        self.define_flux_regions(nfreq=nfreq)

        self._simulation.run(until_after_sources=time_after_source)
        self.flux_freqs = np.array(mp.get_flux_freqs(self._refl_flux))
        self.norm_reflection = np.array(mp.get_fluxes(self._refl_flux))
        self.norm_transmission = np.array(mp.get_fluxes(self._trans_flux))
        self._simulation.save_flux('refl-flux', self._refl_flux)
        self.end_simulation()

        self.define_cell()
        self.define_geometry(norm=False)
        self.define_sources(fcen=fcen, df=df)
        self.define_simulation(res=res)
        self.define_flux_regions(nfreq=nfreq)
        self._simulation.load_minus_flux('refl-flux', self._refl_flux)

        self._simulation.run(until_after_sources=time_after_source)
        self.flux_reflection = np.array(mp.get_fluxes(self._refl_flux))
        self.flux_transmission = np.array(mp.get_fluxes(self._trans_flux))
        self.end_simulation()

        os.system('rm refl-flux.h5')

    def get_transmission(self, fcen=1.27389, df=0.1, res=32, time_after_source=200):
        self.get_spectra(fcen=fcen, df=df, res=res, nfreq=1, time_after_source=time_after_source)
        return self.flux_transmission[0] / self.norm_transmission[0]


class MicroPillarCavity:

    def __init__(self, n_ac, dbr_periodicity, dbr_fill_factor, top_dbr_number, bottom_dbr_number,
                 cavity_length, cavity_radius):
        now = datetime.datetime.now()
        self.directory = 'log/' + now.strftime("%Y-%m-%d") + '/'

        self.n_Ac = n_ac
        self.dbr_periodicity = dbr_periodicity
        self.dbr_fill_factor = dbr_fill_factor
        self.top_dbr_number = top_dbr_number
        self.bottom_dbr_number = bottom_dbr_number
        self.cavity_length = cavity_length
        self.pillar_radius = cavity_radius

        self._cell = None
        self._dimensions = mp.CYLINDRICAL
        self._geometry = None
        self._sources = None
        self._simulation = None

    def create_directory(self):
        system_command('mkdir -p ' + self.directory)

    def define_cell(self, **kwargs):
        pad_r = kwargs.get('pad_r', None)
        if pad_r is None:
            self.pad_r = 5
        else:
            self.pad_r = pad_r
        pad_z = kwargs.get('pad_z', None)
        if pad_z is None:
            self.pad_z = 1
        else:
            self.pad_z = pad_z
        dpml = kwargs.get('dpml', None)
        if dpml is None:
            self.dpml = 1
        else:
            self.dpml = dpml

        self.sr = self.pillar_radius + self.pad_r + self.dpml  # radial size (cell is from 0 to sr)
        self.sz = self.dbr_periodicity * (self.top_dbr_number + self.bottom_dbr_number) + self.cavity_length + 2 * (
                    self.pad_z + self.dpml)
        self._cell = mp.Vector3(self.sr, 0, self.sz)

    def define_geometry(self, norm=False):
        if self._cell is not None:
            self.norm = norm
            # define geometry
            self._geometry = []

            if norm:
                self.cavity_middle_position = 0

            else:
                # from bottom
                # Oxide base layer
                current_layer_position = - self.sz / 2
                current_layer_thickness = self.pad_z + self.dpml
                self._geometry.append(mp.Block(center=mp.Vector3(self.sr / 2, 0,
                                                                 current_layer_position + current_layer_thickness / 2),
                                               size=mp.Vector3(self.sr, 1e20, current_layer_thickness),
                                               material=mp.Medium(index=n_low)))
                current_layer_position += current_layer_thickness

                # Nitride / Oxide pairs
                for i in range(self.bottom_dbr_number):
                    current_layer_thickness = self.dbr_periodicity * self.dbr_fill_factor
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(self.pillar_radius / 2, 0,
                                                   current_layer_position + current_layer_thickness / 2),
                                 size=mp.Vector3(self.pillar_radius, 1e20, current_layer_thickness),
                                 material=mp.Medium(index=n_high)))
                    current_layer_position += current_layer_thickness

                    current_layer_thickness = self.dbr_periodicity * (1 - self.dbr_fill_factor)
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(self.pillar_radius / 2, 0,
                                                   current_layer_position + current_layer_thickness / 2),
                                 size=mp.Vector3(self.pillar_radius, 1e20, current_layer_thickness),
                                 material=mp.Medium(index=n_low)))
                    current_layer_position += current_layer_thickness

                # Cavity
                current_layer_thickness = self.cavity_length
                self._geometry.append(
                    mp.Block(center=mp.Vector3(self.pillar_radius / 2, 0,
                                               current_layer_position + current_layer_thickness / 2),
                             size=mp.Vector3(self.pillar_radius, 1e20, current_layer_thickness),
                             material=mp.Medium(index=self.n_Ac)))
                self.cavity_middle_position = current_layer_position + current_layer_thickness / 2
                current_layer_position += current_layer_thickness

                # Oxide / Nitride pairs
                for i in range(self.top_dbr_number):
                    current_layer_thickness = self.dbr_periodicity * (1 - self.dbr_fill_factor)
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(self.pillar_radius / 2, 0,
                                                   current_layer_position + current_layer_thickness / 2),
                                 size=mp.Vector3(self.pillar_radius, 1e20, current_layer_thickness),
                                 material=mp.Medium(index=n_low)))
                    current_layer_position += current_layer_thickness

                    current_layer_thickness = self.dbr_periodicity * self.dbr_fill_factor
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(self.pillar_radius / 2, 0,
                                                   current_layer_position + current_layer_thickness / 2),
                                 size=mp.Vector3(self.pillar_radius, 1e20, current_layer_thickness),
                                 material=mp.Medium(index=n_high)))
                    current_layer_position += current_layer_thickness
        else:
            raise Exception('Need a cell before defining geometry')

    def define_tuning_geometry(self, tuning_etch=0):
        if self._cell is not None:
            self.tuning_etch = tuning_etch
            self.norm = False
            # define geometry
            self._geometry = []

            # from bottom
            # Oxide base layer
            current_layer_position = - self.sz / 2
            current_layer_thickness = self.pad_z + self.dpml
            self._geometry.append(mp.Block(center=mp.Vector3(self.sr / 2, 0,
                                                             current_layer_position + current_layer_thickness / 2),
                                           size=mp.Vector3(self.sr, 1e20, current_layer_thickness),
                                           material=mp.Medium(index=n_low)))
            current_layer_position += current_layer_thickness

            # Nitride / Oxide pairs
            for i in range(self.bottom_dbr_number):
                current_layer_thickness = self.dbr_periodicity * self.dbr_fill_factor
                self._geometry.append(
                    mp.Block(center=mp.Vector3(self.pillar_radius / 2, 0,
                                               current_layer_position + current_layer_thickness / 2),
                             size=mp.Vector3(self.pillar_radius, 1e20, current_layer_thickness),
                             material=mp.Medium(index=n_high)))
                current_layer_position += current_layer_thickness

                if i == (self.bottom_dbr_number - 1):
                    current_layer_thickness = self.dbr_periodicity * (1 - self.dbr_fill_factor) + tuning_etch
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(self.pillar_radius / 2, 0,
                                                   current_layer_position + current_layer_thickness / 2),
                                 size=mp.Vector3(self.pillar_radius, 1e20, current_layer_thickness),
                                 material=mp.Medium(index=n_low)))
                    current_layer_position += current_layer_thickness

                else:
                    current_layer_thickness = self.dbr_periodicity * (1 - self.dbr_fill_factor)
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(self.pillar_radius / 2, 0,
                                                   current_layer_position + current_layer_thickness / 2),
                                 size=mp.Vector3(self.pillar_radius, 1e20, current_layer_thickness),
                                 material=mp.Medium(index=n_low)))
                    current_layer_position += current_layer_thickness

            # Cavity
            current_layer_thickness = self.cavity_length - 2 * tuning_etch
            self._geometry.append(
                mp.Block(center=mp.Vector3(self.pillar_radius / 2, 0,
                                           current_layer_position + current_layer_thickness / 2),
                         size=mp.Vector3(self.pillar_radius, 1e20, current_layer_thickness),
                         material=mp.Medium(index=self.n_Ac)))
            self.cavity_middle_position = current_layer_position + current_layer_thickness / 2
            current_layer_position += current_layer_thickness

            # Oxide / Nitride pairs
            for i in range(self.top_dbr_number):

                if i == 0:
                    current_layer_thickness = self.dbr_periodicity * (1 - self.dbr_fill_factor) + tuning_etch
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(self.pillar_radius / 2, 0,
                                                   current_layer_position + current_layer_thickness / 2),
                                 size=mp.Vector3(self.pillar_radius, 1e20, current_layer_thickness),
                                 material=mp.Medium(index=n_low)))
                    current_layer_position += current_layer_thickness

                else:
                    current_layer_thickness = self.dbr_periodicity * (1 - self.dbr_fill_factor)
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(self.pillar_radius / 2, 0,
                                                   current_layer_position + current_layer_thickness / 2),
                                 size=mp.Vector3(self.pillar_radius, 1e20, current_layer_thickness),
                                 material=mp.Medium(index=n_low)))
                    current_layer_position += current_layer_thickness

                current_layer_thickness = self.dbr_periodicity * self.dbr_fill_factor
                self._geometry.append(
                    mp.Block(center=mp.Vector3(self.pillar_radius / 2, 0,
                                               current_layer_position + current_layer_thickness / 2),
                             size=mp.Vector3(self.pillar_radius, 1e20, current_layer_thickness),
                             material=mp.Medium(index=n_high)))
                current_layer_position += current_layer_thickness
        else:
            raise Exception('Need a cell before defining geometry')

    def define_point_sources(self, fcen=1.27389, df=0.1):
        if self._geometry is not None:
            self.fcen = fcen
            self.df = df
            self._sources = [mp.Source(src=mp.GaussianSource(self.fcen, fwidth=self.df),
                                       component=mp.Er,
                                       center=mp.Vector3(0, 0, self.cavity_middle_position))]
        else:
            raise Exception('Need a geometry before defining sources')

    def define_line_sources(self, fcen=1.27389, df=0.1):
        if self._geometry is not None:
            self.fcen = fcen
            self.df = df
            self._sources = [mp.Source(src=mp.GaussianSource(self.fcen, fwidth=self.df),
                                       component=mp.Er,
                                       center=mp.Vector3(0, 0, self.cavity_middle_position),
                                       size=mp.Vector3(self.pillar_radius * 2, 0, 0))]
        else:
            raise Exception('Need a geometry before defining sources')

    def define_simulation(self, res=32):
        if (self._cell is not None) & (self._geometry is not None) & (self._sources is not None):
            self.res = res
            m = 1  # define integer for cylindrical symmetry
            pml_layers = [mp.PML(self.dpml)]

            if self.norm:
                self._simulation = mp.Simulation(cell_size=self._cell,
                                                 geometry=self._geometry,
                                                 boundary_layers=pml_layers,
                                                 resolution=res,
                                                 sources=self._sources,
                                                 dimensions=self._dimensions,
                                                 default_material=mp.Medium(index=self.n_Ac),
                                                 m=m)
            else:
                self._simulation = mp.Simulation(cell_size=self._cell,
                                                 geometry=self._geometry,
                                                 boundary_layers=pml_layers,
                                                 resolution=res,
                                                 sources=self._sources,
                                                 dimensions=self._dimensions,
                                                 m=m)
        else:
            raise Exception('Environment not fully defined. Need a cell, geometry and sources')

    def end_simulation(self):
        self._simulation.reset_meep()
        self._cell = None
        self._geometry = None
        self._sources = None
        self._simulation = None

    def get_ldos(self, time_after_source=200, nfreq=100):
        if self._simulation is not None:
            self.time_after_source = time_after_source
            self._simulation.run(mp.dft_ldos(self.fcen, self.df, nfreq), until_after_sources=self.time_after_source)

            mp.all_wait()
            if mp.am_master():
                ldos_instance = mp._dft_ldos(self.fcen - self.df / 2, self.fcen + self.df / 2, nfreq)
                self.ldos_results = np.transpose(
                    np.array([mp.get_ldos_freqs(ldos_instance), self._simulation.ldos_data]))

        else:
            raise Exception('Cannot run null simulation')

    def get_qs(self, time_after_source=200):
        if self._simulation is not None:
            self.time_after_source = time_after_source

            harminv_instance = mp.Harminv(mp.Er,
                                          mp.Vector3(0, 0, self.cavity_middle_position),
                                          self.fcen, self.df)

            self._simulation.run(mp.after_sources(harminv_instance),
                                 until_after_sources=self.time_after_source)

            mp.all_wait()
            if mp.am_master():
                self.q_results = []
                for mode in harminv_instance.modes:
                    self.q_results.append([mode.freq, mode.decay, mode.Q, abs(mode.amp)])

                self.q_results = np.array(self.q_results)

                if len(self.q_results.shape) == 1:
                    self.q_results = np.array([[self.fcen, 0, 0, 0]])

                self.q_results = self.q_results[self.q_results[:, 2] > 0]

                if self.q_results.shape[0] == 0:
                    self.q_results = np.array([[self.fcen, 0, 0, 0]])

                self.q_results = self.q_results[abs(self.q_results[:, 0] - self.fcen) < (self.df / 2)]

                if self.q_results.shape[0] == 0:
                    self.q_results = np.array([[self.fcen, 0, 0, 0]])

                return self.q_results[:, 2]
        else:
            raise Exception('Cannot run null simulation')

    def get_mode_volume(self, time_after_source=200):
        if self._simulation is not None:
            self.time_after_source = time_after_source
            self._simulation.run(until_after_sources=self.time_after_source)
            computational_cell = self._simulation.fields.total_volume()
            modal_volume = self._simulation.fields.modal_volume_in_box(where=computational_cell)
            mp.all_wait()
            if mp.am_master():
                self.modal_volume_result = modal_volume
                return modal_volume

        else:
            raise Exception('Cannot run null simulation')


class GaussianDefectCavity:

    def __init__(self, n_ac, dbr_periodicity, dbr_fill_factor, top_dbr_number, bottom_dbr_number,
                 cavity_length, defect_height, defect_sigma):
        now = datetime.datetime.now()
        self.directory = 'log/' + now.strftime("%Y-%m-%d") + '/'

        self.n_Ac = n_ac
        self.dbr_periodicity = dbr_periodicity
        self.dbr_fill_factor = dbr_fill_factor
        self.top_dbr_number = top_dbr_number
        self.bottom_dbr_number = bottom_dbr_number
        self.cavity_length = cavity_length
        self.defect_height = defect_height
        self.defect_sigma = defect_sigma
        self.defect_resolution = RES

        self._cell = None
        self._dimensions = mp.CYLINDRICAL
        self._geometry = None
        self._sources = None
        self._simulation = None

    def create_directory(self):
        system_command('mkdir -p ' + self.directory)

    def define_cell(self, **kwargs):
        pad_r = kwargs.get('pad_r', None)
        if pad_r is None:
            self.pad_r = 5
        else:
            self.pad_r = pad_r
        pad_z = kwargs.get('pad_z', None)
        if pad_z is None:
            self.pad_z = 1
        else:
            self.pad_z = pad_z
        dpml = kwargs.get('dpml', None)
        if dpml is None:
            self.dpml = 1
        else:
            self.dpml = dpml

        self.sr = self.pad_r + self.dpml  # radial size (cell is from 0 to sr)
        self.sz = self.dbr_periodicity * (self.top_dbr_number + self.bottom_dbr_number) + self.cavity_length + 2 * (
                    self.pad_z + self.dpml)
        self._cell = mp.Vector3(self.sr, 0, self.sz)

    def define_geometry(self, norm=False):
        if self._cell is not None:
            self.norm = norm
            # define geometry
            self._geometry = []

            if norm:
                self.cavity_middle_position = 0

            else:
                # from bottom
                # Oxide base layer
                current_layer_position = - self.sz / 2
                current_layer_thickness = self.pad_z + self.dpml
                self._geometry.append(mp.Block(center=mp.Vector3(0, 0,
                                                                 current_layer_position + current_layer_thickness / 2),
                                               size=mp.Vector3(1e20, 1e20, current_layer_thickness),
                                               material=mp.Medium(index=n_low)))
                current_layer_position += current_layer_thickness

                # Nitride / Oxide pairs
                for i in range(self.bottom_dbr_number):
                    current_layer_thickness = self.dbr_periodicity * self.dbr_fill_factor
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(0, 0, current_layer_position + current_layer_thickness / 2),
                                 size=mp.Vector3(1e20, 1e20, current_layer_thickness),
                                 material=mp.Medium(index=n_high)))
                    current_layer_position += current_layer_thickness

                    current_layer_thickness = self.dbr_periodicity * (1 - self.dbr_fill_factor)
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(0, 0, current_layer_position + current_layer_thickness / 2),
                                 size=mp.Vector3(1e20, 1e20, current_layer_thickness),
                                 material=mp.Medium(index=n_low)))
                    current_layer_position += current_layer_thickness

                # Cavity
                current_layer_thickness = self.cavity_length - self.defect_height
                self._geometry.append(
                    mp.Block(center=mp.Vector3(0, 0, current_layer_position + current_layer_thickness / 2),
                             size=mp.Vector3(1e20, 1e20, current_layer_thickness),
                             material=mp.Medium(index=self.n_Ac)))
                self.cavity_middle_position = current_layer_position + self.cavity_length / 2
                current_layer_position += current_layer_thickness

                # Cavity defect
                quotient, remainder = divmod(self.defect_height, self.defect_resolution)

                for n in range(int(quotient)):
                    current_defect_position = self.defect_resolution / 2 + n * self.defect_resolution
                    outer_waist = 1e20

                    if current_defect_position > self.dbr_periodicity * (1 - self.dbr_fill_factor):
                        self._geometry.append(
                            mp.Block(center=mp.Vector3(outer_waist / 2,
                                                       0,
                                                       current_layer_position + self.defect_resolution / 2),
                                     size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                     material=mp.Medium(index=n_high)))

                        outer_waist = self.defect_sigma * np.sqrt(
                            2 * np.log(self.defect_height / (current_defect_position
                                                             - self.dbr_periodicity * (1 - self.dbr_fill_factor))))

                    self._geometry.append(mp.Block(center=mp.Vector3(outer_waist / 2, 0, current_layer_position + self.defect_resolution / 2),
                             size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                             material=mp.Medium(index=n_low)))

                    waist = self.defect_sigma * np.sqrt(2 * np.log(self.defect_height / current_defect_position))

                    self._geometry.append(
                        mp.Block(center=mp.Vector3(waist / 2, 0, current_layer_position + self.defect_resolution / 2),
                                 size=mp.Vector3(waist, 1e20, self.defect_resolution),
                                 material=mp.Medium(index=self.n_Ac)))

                    current_layer_position += self.defect_resolution

                # Oxide / Nitride pairs
                for i in range(self.top_dbr_number - 1):

                    # Oxide
                    quotient, remainder = divmod(self.dbr_periodicity * (1 - self.dbr_fill_factor),
                                                 self.defect_resolution)

                    for n in range(int(quotient)):
                        current_defect_position = self.defect_height \
                                                  - self.dbr_periodicity * (1 - self.dbr_fill_factor) \
                                                  + self.defect_resolution / 2 + n * self.defect_resolution
                        outer_waist = 1e20

                        if current_defect_position <= 0:
                            self._geometry.append(
                                mp.Block(center=mp.Vector3(outer_waist / 2, 0,
                                                           current_layer_position + self.defect_resolution / 2),
                                         size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                         material=mp.Medium(index=n_low)))

                        else:
                            if current_defect_position > self.dbr_periodicity * self.dbr_fill_factor:
                                self._geometry.append(
                                    mp.Block(center=mp.Vector3(outer_waist / 2, 0,
                                                               current_layer_position + self.defect_resolution / 2),
                                             size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                             material=mp.Medium(index=n_low)))

                                outer_waist = self.defect_sigma * np.sqrt(
                                    2 * np.log(self.defect_height / (current_defect_position
                                                                     - self.dbr_periodicity * self.dbr_fill_factor)))

                            self._geometry.append(mp.Block(
                                center=mp.Vector3(outer_waist / 2, 0, current_layer_position + self.defect_resolution / 2),
                                size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                material=mp.Medium(index=n_high)))

                            waist = self.defect_sigma * np.sqrt(2 * np.log(self.defect_height / current_defect_position))

                            self._geometry.append(
                                mp.Block(
                                    center=mp.Vector3(waist / 2, 0, current_layer_position + self.defect_resolution / 2),
                                    size=mp.Vector3(waist, 1e20, self.defect_resolution),
                                    material=mp.Medium(index=n_low)))

                        current_layer_position += self.defect_resolution

                    # Nitride
                    quotient, remainder = divmod(self.dbr_periodicity * self.dbr_fill_factor,
                                                 self.defect_resolution)

                    for n in range(int(quotient)):
                        current_defect_position = self.defect_height \
                                                  - self.dbr_periodicity * self.dbr_fill_factor \
                                                  + self.defect_resolution / 2 + n * self.defect_resolution
                        outer_waist = 1e20

                        if current_defect_position <= 0:
                            self._geometry.append(
                                mp.Block(center=mp.Vector3(outer_waist / 2, 0,
                                                           current_layer_position + self.defect_resolution / 2),
                                         size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                         material=mp.Medium(index=n_high)))

                        else:
                            if current_defect_position > self.dbr_periodicity * (1 - self.dbr_fill_factor):
                                self._geometry.append(
                                    mp.Block(center=mp.Vector3(outer_waist / 2, 0,
                                                               current_layer_position + self.defect_resolution / 2),
                                             size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                             material=mp.Medium(index=n_high)))

                                outer_waist = self.defect_sigma * np.sqrt(
                                    2 * np.log(self.defect_height / (current_defect_position
                                                                     - self.dbr_periodicity * (1 - self.dbr_fill_factor))))

                            self._geometry.append(mp.Block(
                                center=mp.Vector3(outer_waist / 2, 0,
                                                  current_layer_position + self.defect_resolution / 2),
                                size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                material=mp.Medium(index=n_low)))

                            waist = self.defect_sigma * np.sqrt(
                                2 * np.log(self.defect_height / current_defect_position))

                            self._geometry.append(
                                mp.Block(
                                    center=mp.Vector3(waist / 2, 0,
                                                      current_layer_position + self.defect_resolution / 2),
                                    size=mp.Vector3(waist, 1e20, self.defect_resolution),
                                    material=mp.Medium(index=n_high)))

                        current_layer_position += self.defect_resolution


                # Final layers
                # Oxide
                quotient, remainder = divmod(self.dbr_periodicity * (1 - self.dbr_fill_factor),
                                             self.defect_resolution)

                for n in range(int(quotient)):
                    current_defect_position = self.defect_height \
                                              - self.dbr_periodicity * (1 - self.dbr_fill_factor) \
                                              + self.defect_resolution / 2 + n * self.defect_resolution
                    outer_waist = 1e20

                    if current_defect_position <= 0:
                        self._geometry.append(
                            mp.Block(center=mp.Vector3(outer_waist / 2, 0,
                                                       current_layer_position + self.defect_resolution / 2),
                                     size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                     material=mp.Medium(index=n_low)))

                    else:
                        if current_defect_position > self.dbr_periodicity * self.dbr_fill_factor:
                            outer_waist = self.defect_sigma * np.sqrt(
                                2 * np.log(self.defect_height / (current_defect_position
                                                                 - self.dbr_periodicity * self.dbr_fill_factor)))

                        self._geometry.append(mp.Block(
                            center=mp.Vector3(outer_waist / 2, 0,
                                              current_layer_position + self.defect_resolution / 2),
                            size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                            material=mp.Medium(index=n_high)))

                        waist = self.defect_sigma * np.sqrt(
                            2 * np.log(self.defect_height / current_defect_position))

                        self._geometry.append(
                            mp.Block(
                                center=mp.Vector3(waist / 2, 0,
                                                  current_layer_position + self.defect_resolution / 2),
                                size=mp.Vector3(waist, 1e20, self.defect_resolution),
                                material=mp.Medium(index=n_low)))

                    current_layer_position += self.defect_resolution

                # Nitride
                quotient, remainder = divmod(self.dbr_periodicity * self.dbr_fill_factor,
                                             self.defect_resolution)

                for n in range(int(quotient)):
                    current_defect_position = self.defect_height \
                                              - self.dbr_periodicity * self.dbr_fill_factor \
                                              + self.defect_resolution / 2 + n * self.defect_resolution
                    outer_waist = 1e20

                    if current_defect_position <= 0:
                        self._geometry.append(
                            mp.Block(center=mp.Vector3(outer_waist / 2, 0,
                                                       current_layer_position + self.defect_resolution / 2),
                                     size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                     material=mp.Medium(index=n_high)))

                    else:
                        waist = self.defect_sigma * np.sqrt(
                            2 * np.log(self.defect_height / current_defect_position))

                        self._geometry.append(
                            mp.Block(
                                center=mp.Vector3(waist / 2, 0,
                                                  current_layer_position + self.defect_resolution / 2),
                                size=mp.Vector3(waist, 1e20, self.defect_resolution),
                                material=mp.Medium(index=n_high)))

                    current_layer_position += self.defect_resolution

        else:
            raise Exception('Need a cell before defining geometry')

    def define_tuning_geometry(self, tuning_etch=0):
        if self._cell is not None:
            # define geometry
            self.norm = False
            self.tuning_etch = tuning_etch
            self._geometry = []

            # from bottom
            # Oxide base layer
            current_layer_position = - self.sz / 2
            current_layer_thickness = self.pad_z + self.dpml
            self._geometry.append(mp.Block(center=mp.Vector3(0, 0,
                                                             current_layer_position + current_layer_thickness / 2),
                                           size=mp.Vector3(1e20, 1e20, current_layer_thickness),
                                           material=mp.Medium(index=n_low)))
            current_layer_position += current_layer_thickness

            # Nitride / Oxide pairs
            for i in range(self.bottom_dbr_number):
                current_layer_thickness = self.dbr_periodicity * self.dbr_fill_factor
                self._geometry.append(
                    mp.Block(center=mp.Vector3(0, 0, current_layer_position + current_layer_thickness / 2),
                             size=mp.Vector3(1e20, 1e20, current_layer_thickness),
                             material=mp.Medium(index=n_high)))
                current_layer_position += current_layer_thickness

                if i == (self.bottom_dbr_number - 1):
                    current_layer_thickness = self.dbr_periodicity * (1 - self.dbr_fill_factor) + tuning_etch
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(0, 0, current_layer_position + current_layer_thickness / 2),
                                 size=mp.Vector3(1e20, 1e20, current_layer_thickness),
                                 material=mp.Medium(index=n_low)))
                    current_layer_position += current_layer_thickness
                else:
                    current_layer_thickness = self.dbr_periodicity * (1 - self.dbr_fill_factor)
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(0, 0, current_layer_position + current_layer_thickness / 2),
                                 size=mp.Vector3(1e20, 1e20, current_layer_thickness),
                                 material=mp.Medium(index=n_low)))
                    current_layer_position += current_layer_thickness

            # Cavity
            current_layer_thickness = self.cavity_length - self.defect_height - 2 * tuning_etch
            self._geometry.append(
                mp.Block(center=mp.Vector3(0, 0, current_layer_position + current_layer_thickness / 2),
                         size=mp.Vector3(1e20, 1e20, current_layer_thickness),
                         material=mp.Medium(index=self.n_Ac)))
            self.cavity_middle_position = current_layer_position + self.cavity_length / 2
            current_layer_position += current_layer_thickness

            # Cavity defect
            quotient, remainder = divmod(self.defect_height, self.defect_resolution)

            for n in range(int(quotient)):
                current_defect_position = self.defect_resolution / 2 + n * self.defect_resolution
                outer_waist = 1e20

                if current_defect_position > self.dbr_periodicity * (1 - self.dbr_fill_factor):
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(outer_waist / 2,
                                                   0,
                                                   current_layer_position + self.defect_resolution / 2),
                                 size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                 material=mp.Medium(index=n_high)))

                    outer_waist = self.defect_sigma * np.sqrt(
                        2 * np.log(self.defect_height / (current_defect_position
                                                         - self.dbr_periodicity * (1 - self.dbr_fill_factor))))

                self._geometry.append(mp.Block(center=mp.Vector3(outer_waist / 2, 0,
                                                                 current_layer_position + self.defect_resolution / 2),
                         size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                         material=mp.Medium(index=n_low)))

                waist = self.defect_sigma * np.sqrt(2 * np.log(self.defect_height / current_defect_position))

                self._geometry.append(
                    mp.Block(center=mp.Vector3(waist / 2, 0, current_layer_position + self.defect_resolution / 2),
                             size=mp.Vector3(waist, 1e20, self.defect_resolution),
                             material=mp.Medium(index=self.n_Ac)))

                current_layer_position += self.defect_resolution

            # Oxide / Nitride pairs
            for i in range(self.top_dbr_number - 1):

                # Oxide
                if i == 0:
                    quotient, remainder = divmod(self.dbr_periodicity * (1 - self.dbr_fill_factor) + tuning_etch,
                                                 self.defect_resolution)

                    for n in range(int(quotient)):
                        current_defect_position = self.defect_height \
                                                  - (self.dbr_periodicity * (1 - self.dbr_fill_factor) + tuning_etch) \
                                                  + self.defect_resolution / 2 + n * self.defect_resolution
                        outer_waist = 1e20

                        if current_defect_position <= 0:
                            self._geometry.append(
                                mp.Block(center=mp.Vector3(outer_waist / 2, 0,
                                                           current_layer_position + self.defect_resolution / 2),
                                         size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                         material=mp.Medium(index=n_low)))

                        else:
                            if current_defect_position > self.dbr_periodicity * self.dbr_fill_factor:
                                self._geometry.append(
                                    mp.Block(center=mp.Vector3(outer_waist / 2, 0,
                                                               current_layer_position + self.defect_resolution / 2),
                                             size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                             material=mp.Medium(index=n_low)))

                                outer_waist = self.defect_sigma * np.sqrt(
                                    2 * np.log(self.defect_height / (current_defect_position
                                                                     - self.dbr_periodicity * self.dbr_fill_factor)))

                            self._geometry.append(mp.Block(
                                center=mp.Vector3(outer_waist / 2, 0,
                                                  current_layer_position + self.defect_resolution / 2),
                                size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                material=mp.Medium(index=n_high)))

                            waist = self.defect_sigma * np.sqrt(
                                2 * np.log(self.defect_height / current_defect_position))

                            self._geometry.append(
                                mp.Block(
                                    center=mp.Vector3(waist / 2, 0,
                                                      current_layer_position + self.defect_resolution / 2),
                                    size=mp.Vector3(waist, 1e20, self.defect_resolution),
                                    material=mp.Medium(index=n_low)))

                        current_layer_position += self.defect_resolution

                else:
                    quotient, remainder = divmod(self.dbr_periodicity * (1 - self.dbr_fill_factor),
                                                 self.defect_resolution)

                    for n in range(int(quotient)):
                        current_defect_position = self.defect_height \
                                                  - self.dbr_periodicity * (1 - self.dbr_fill_factor) \
                                                  + self.defect_resolution / 2 + n * self.defect_resolution
                        outer_waist = 1e20

                        if current_defect_position <= 0:
                            self._geometry.append(
                                mp.Block(center=mp.Vector3(outer_waist / 2, 0,
                                                           current_layer_position + self.defect_resolution / 2),
                                         size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                         material=mp.Medium(index=n_low)))

                        else:
                            if current_defect_position > self.dbr_periodicity * self.dbr_fill_factor:
                                self._geometry.append(
                                    mp.Block(center=mp.Vector3(outer_waist / 2, 0,
                                                               current_layer_position + self.defect_resolution / 2),
                                             size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                             material=mp.Medium(index=n_low)))

                                outer_waist = self.defect_sigma * np.sqrt(
                                    2 * np.log(self.defect_height / (current_defect_position
                                                                     - self.dbr_periodicity * self.dbr_fill_factor)))

                            self._geometry.append(mp.Block(
                                center=mp.Vector3(outer_waist / 2, 0,
                                                  current_layer_position + self.defect_resolution / 2),
                                size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                material=mp.Medium(index=n_high)))

                            waist = self.defect_sigma * np.sqrt(
                                2 * np.log(self.defect_height / current_defect_position))

                            self._geometry.append(
                                mp.Block(
                                    center=mp.Vector3(waist / 2, 0,
                                                      current_layer_position + self.defect_resolution / 2),
                                    size=mp.Vector3(waist, 1e20, self.defect_resolution),
                                    material=mp.Medium(index=n_low)))

                        current_layer_position += self.defect_resolution

                # Nitride
                quotient, remainder = divmod(self.dbr_periodicity * self.dbr_fill_factor,
                                             self.defect_resolution)

                for n in range(int(quotient)):
                    current_defect_position = self.defect_height \
                                              - self.dbr_periodicity * self.dbr_fill_factor \
                                              + self.defect_resolution / 2 + n * self.defect_resolution
                    outer_waist = 1e20

                    if current_defect_position <= 0:
                        self._geometry.append(
                            mp.Block(center=mp.Vector3(outer_waist / 2, 0,
                                                       current_layer_position + self.defect_resolution / 2),
                                     size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                     material=mp.Medium(index=n_high)))

                    else:
                        if current_defect_position > self.dbr_periodicity * (1 - self.dbr_fill_factor):
                            self._geometry.append(
                                mp.Block(center=mp.Vector3(outer_waist / 2, 0,
                                                           current_layer_position + self.defect_resolution / 2),
                                         size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                         material=mp.Medium(index=n_high)))

                            outer_waist = self.defect_sigma * np.sqrt(
                                2 * np.log(self.defect_height / (current_defect_position
                                                                 - self.dbr_periodicity * (1 - self.dbr_fill_factor))))

                        self._geometry.append(mp.Block(
                            center=mp.Vector3(outer_waist / 2, 0,
                                              current_layer_position + self.defect_resolution / 2),
                            size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                            material=mp.Medium(index=n_low)))

                        waist = self.defect_sigma * np.sqrt(
                            2 * np.log(self.defect_height / current_defect_position))

                        self._geometry.append(
                            mp.Block(
                                center=mp.Vector3(waist / 2, 0,
                                                  current_layer_position + self.defect_resolution / 2),
                                size=mp.Vector3(waist, 1e20, self.defect_resolution),
                                material=mp.Medium(index=n_high)))

                    current_layer_position += self.defect_resolution


            # Final layers
            # Oxide
            quotient, remainder = divmod(self.dbr_periodicity * (1 - self.dbr_fill_factor),
                                         self.defect_resolution)

            for n in range(int(quotient)):
                current_defect_position = self.defect_height \
                                          - self.dbr_periodicity * (1 - self.dbr_fill_factor) \
                                          + self.defect_resolution / 2 + n * self.defect_resolution
                outer_waist = 1e20

                if current_defect_position <= 0:
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(outer_waist / 2, 0,
                                                   current_layer_position + self.defect_resolution / 2),
                                 size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                 material=mp.Medium(index=n_low)))

                else:
                    if current_defect_position > self.dbr_periodicity * self.dbr_fill_factor:
                        outer_waist = self.defect_sigma * np.sqrt(
                            2 * np.log(self.defect_height / (current_defect_position
                                                             - self.dbr_periodicity * self.dbr_fill_factor)))

                    self._geometry.append(mp.Block(
                        center=mp.Vector3(outer_waist / 2, 0,
                                          current_layer_position + self.defect_resolution / 2),
                        size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                        material=mp.Medium(index=n_high)))

                    waist = self.defect_sigma * np.sqrt(
                        2 * np.log(self.defect_height / current_defect_position))

                    self._geometry.append(
                        mp.Block(
                            center=mp.Vector3(waist / 2, 0,
                                              current_layer_position + self.defect_resolution / 2),
                            size=mp.Vector3(waist, 1e20, self.defect_resolution),
                            material=mp.Medium(index=n_low)))

                current_layer_position += self.defect_resolution

            # Nitride
            quotient, remainder = divmod(self.dbr_periodicity * self.dbr_fill_factor,
                                         self.defect_resolution)

            for n in range(int(quotient)):
                current_defect_position = self.defect_height \
                                          - self.dbr_periodicity * self.dbr_fill_factor \
                                          + self.defect_resolution / 2 + n * self.defect_resolution
                outer_waist = 1e20

                if current_defect_position <= 0:
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(outer_waist / 2, 0,
                                                   current_layer_position + self.defect_resolution / 2),
                                 size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                 material=mp.Medium(index=n_high)))

                else:
                    waist = self.defect_sigma * np.sqrt(
                        2 * np.log(self.defect_height / current_defect_position))

                    self._geometry.append(
                        mp.Block(
                            center=mp.Vector3(waist / 2, 0,
                                              current_layer_position + self.defect_resolution / 2),
                            size=mp.Vector3(waist, 1e20, self.defect_resolution),
                            material=mp.Medium(index=n_high)))

                current_layer_position += self.defect_resolution

        else:
            raise Exception('Need a cell before defining geometry')

    def define_electrode_geometry(self, buffer_thickness, electrode_thickness):
        if self._cell is not None:

            # define geometry
            self.norm = False
            self.buffer_thickness = buffer_thickness
            self.electrode_thickness = electrode_thickness
            self.sz += 2 * (buffer_thickness + electrode_thickness)
            self._cell = mp.Vector3(self.sr, 0, self.sz)
            self._geometry = []

            #region Bottom half cavity

            # from bottom
            # Oxide base layer
            current_layer_position = - self.sz / 2
            current_layer_thickness = self.pad_z + self.dpml
            self._geometry.append(mp.Block(center=mp.Vector3(0, 0,
                                                             current_layer_position + current_layer_thickness / 2),
                                           size=mp.Vector3(1e20, 1e20, current_layer_thickness),
                                           material=mp.Medium(index=n_low)))
            current_layer_position += current_layer_thickness

            # Nitride / Oxide pairs
            for i in range(self.bottom_dbr_number):
                current_layer_thickness = self.dbr_periodicity * self.dbr_fill_factor
                self._geometry.append(
                    mp.Block(center=mp.Vector3(0, 0, current_layer_position + current_layer_thickness / 2),
                             size=mp.Vector3(1e20, 1e20, current_layer_thickness),
                             material=mp.Medium(index=n_high)))
                current_layer_position += current_layer_thickness

                current_layer_thickness = self.dbr_periodicity * (1 - self.dbr_fill_factor)
                self._geometry.append(
                    mp.Block(center=mp.Vector3(0, 0, current_layer_position + current_layer_thickness / 2),
                             size=mp.Vector3(1e20, 1e20, current_layer_thickness),
                             material=mp.Medium(index=n_low)))
                current_layer_position += current_layer_thickness

            # Electrode layer
            current_layer_thickness = self.electrode_thickness
            self._geometry.append(
                mp.Block(center=mp.Vector3(0, 0, current_layer_position + current_layer_thickness / 2),
                         size=mp.Vector3(1e20, 1e20, current_layer_thickness),
                         material=mp.Medium(index=n_electrode)))
            current_layer_position += current_layer_thickness

            # Buffer layer
            current_layer_thickness = self.buffer_thickness
            self._geometry.append(
                mp.Block(center=mp.Vector3(0, 0, current_layer_position + current_layer_thickness / 2),
                         size=mp.Vector3(1e20, 1e20, current_layer_thickness),
                         material=mp.Medium(index=n_low)))
            current_layer_position += current_layer_thickness

            # Cavity
            current_layer_thickness = self.cavity_length - self.defect_height
            self._geometry.append(
                mp.Block(center=mp.Vector3(0, 0, current_layer_position + current_layer_thickness / 2),
                         size=mp.Vector3(1e20, 1e20, current_layer_thickness),
                         material=mp.Medium(index=self.n_Ac)))
            self.cavity_middle_position = current_layer_position + self.cavity_length / 2
            current_layer_position += current_layer_thickness

            #endregion

            #region Top half cavity

            layer_queue = []; layer_distances = []
            layer_queue.append('cavity'); layer_distances.append(self.defect_height)
            layer_queue.append('low'); layer_distances.append(layer_distances[-1] + self.buffer_thickness)
            layer_queue.append('electrode'); layer_distances.append(layer_distances[-1] + self.electrode_thickness)
            for i in range(self.top_dbr_number):
                layer_queue.append('low'); layer_distances.append(layer_distances[-1] +
                                                                  self.dbr_periodicity * (1 - self.dbr_fill_factor))
                layer_queue.append('high'); layer_distances.append(layer_distances[-1] +
                                                                   self.dbr_periodicity * self.dbr_fill_factor)
            layer_queue.append('low'); layer_distances.append(layer_distances[-1] + 2)
            layer_queue.append('end'); layer_distances.append(1e20)

            for i in range(len(layer_queue)-1):

                current_layer_thickness = layer_distances[0]
                quotient, remainder = divmod(current_layer_thickness, self.defect_resolution)
                layer_distances = [i - current_layer_thickness for i in layer_distances]
                current_defect_position = self.defect_height - current_layer_thickness - self.defect_resolution / 2

                for n in range(int(quotient)):

                    current_defect_position += self.defect_resolution
                    outer_waist = 1e20

                    if current_defect_position <= 0:

                        if layer_queue[0] == 'cavity':
                            self._geometry.append(
                                mp.Block(center=mp.Vector3(outer_waist / 2,
                                                           0,
                                                           current_layer_position + self.defect_resolution / 2),
                                         size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                         material=mp.Medium(index=self.n_Ac)))

                        elif layer_queue[0] == 'low':
                            self._geometry.append(
                                mp.Block(center=mp.Vector3(outer_waist / 2,
                                                           0,
                                                           current_layer_position + self.defect_resolution / 2),
                                         size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                         material=mp.Medium(index=n_low)))

                        elif layer_queue[0] == 'high':
                            self._geometry.append(
                                mp.Block(center=mp.Vector3(outer_waist / 2,
                                                           0,
                                                           current_layer_position + self.defect_resolution / 2),
                                         size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                         material=mp.Medium(index=n_high)))

                        elif layer_queue[0] == 'electrode':
                            self._geometry.append(
                                mp.Block(center=mp.Vector3(outer_waist / 2,
                                                           0,
                                                           current_layer_position + self.defect_resolution / 2),
                                         size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                         material=mp.Medium(index=n_electrode)))

                        current_layer_position += self.defect_resolution

                        continue

                    for m in range(len(layer_queue)):

                        if current_defect_position < layer_distances[m]:

                            for l in range(m + 1):

                                if l != 0:
                                    outer_waist = self.defect_sigma * np.sqrt(
                                        2 * np.log(
                                            self.defect_height / (current_defect_position - layer_distances[m - l])))

                                if layer_queue[m - l] == 'cavity':
                                    self._geometry.append(
                                        mp.Block(center=mp.Vector3(outer_waist / 2,
                                                                   0,
                                                                   current_layer_position + self.defect_resolution / 2),
                                                 size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                                 material=mp.Medium(index=self.n_Ac)))

                                elif layer_queue[m - l] == 'low':
                                    self._geometry.append(
                                        mp.Block(center=mp.Vector3(outer_waist / 2,
                                                                   0,
                                                                   current_layer_position + self.defect_resolution / 2),
                                                 size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                                 material=mp.Medium(index=n_low)))

                                elif layer_queue[m - l] == 'high':
                                    self._geometry.append(
                                        mp.Block(center=mp.Vector3(outer_waist / 2,
                                                                   0,
                                                                   current_layer_position + self.defect_resolution / 2),
                                                 size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                                 material=mp.Medium(index=n_high)))

                                elif layer_queue[m - l] == 'electrode':
                                    self._geometry.append(
                                        mp.Block(center=mp.Vector3(outer_waist / 2,
                                                                   0,
                                                                   current_layer_position + self.defect_resolution / 2),
                                                 size=mp.Vector3(outer_waist, 1e20, self.defect_resolution),
                                                 material=mp.Medium(index=n_electrode)))

                            break

                    current_layer_position += self.defect_resolution

                layer_queue.pop(0); layer_distances.pop(0)



            #endregion

        else:
            raise Exception('Need a cell before defining geometry')

    def define_point_sources(self, fcen=1.27389, df=0.1):
        if self._geometry is not None:
            self.fcen = fcen
            self.df = df
            self._sources = [mp.Source(src=mp.GaussianSource(self.fcen, fwidth=self.df),
                                       component=mp.Er,
                                       center=mp.Vector3(0, 0, self.cavity_middle_position))]
        else:
            raise Exception('Need a geometry before defining sources')

    def define_line_sources(self, fcen=1.27389, df=0.1):
        if self._geometry is not None:
            self.fcen = fcen
            self.df = df
            self._sources = [mp.Source(src=mp.GaussianSource(self.fcen, fwidth=self.df),
                                       component=mp.Er,
                                       center=mp.Vector3(self.defect_sigma / 4, 0, self.cavity_middle_position),
                                       size=mp.Vector3(self.defect_sigma / 2, 0, 0))]
        else:
            raise Exception('Need a geometry before defining sources')

    def define_simulation(self, res=32):
        if (self._cell is not None) & (self._geometry is not None) & (self._sources is not None):
            self.res = res
            m = 1  # define integer for cylindrical symmetry
            pml_layers = [mp.PML(self.dpml)]

            if self.norm:
                self._simulation = mp.Simulation(cell_size=self._cell,
                                                 geometry=self._geometry,
                                                 boundary_layers=pml_layers,
                                                 resolution=res,
                                                 sources=self._sources,
                                                 dimensions=self._dimensions,
                                                 default_material=mp.Medium(index=self.n_Ac),
                                                 m=m,
                                                 progress_interval=60,
                                                 eps_averaging=False)
            else:
                self._simulation = mp.Simulation(cell_size=self._cell,
                                                 geometry=self._geometry,
                                                 boundary_layers=pml_layers,
                                                 resolution=res,
                                                 sources=self._sources,
                                                 dimensions=self._dimensions,
                                                 m=m,
                                                 progress_interval=60,
                                                 eps_averaging=False)
        else:
            raise Exception('Environment not fully defined. Need a cell, geometry and sources')

    def end_simulation(self):
        self._simulation.reset_meep()
        self._cell = None
        self._geometry = None
        self._sources = None
        self._simulation = None

    def get_epsilon(self):
        if self._simulation is not None:
            self._simulation.init_fields()
            eps_data = self._simulation.get_array(mp.Vector3(), mp.Vector3(2 * self.sr, 0, self.sz),
                                                  mp.Dielectric)[::-1]
            return eps_data

    def get_ldos(self, time_after_source=200, nfreq=100):
        if self._simulation is not None:
            self.time_after_source = time_after_source
            self._simulation.run(mp.dft_ldos(self.fcen, self.df, nfreq), until_after_sources=self.time_after_source)
            mp.all_wait()
            if mp.am_master():
                ldos_instance = mp._dft_ldos(self.fcen - self.df / 2, self.fcen + self.df / 2, nfreq)
                self.ldos_results = np.transpose(
                    np.array([mp.get_ldos_freqs(ldos_instance), self._simulation.ldos_data]))

        else:
            raise Exception('Cannot run null simulation')

    def get_qs(self, time_after_source=200):
        if self._simulation is not None:
            self.time_after_source = time_after_source

            harminv_instance = mp.Harminv(mp.Er,
                                          mp.Vector3(0, 0, self.cavity_middle_position),
                                          self.fcen, self.df)

            self._simulation.run(mp.after_sources(harminv_instance),
                                 until_after_sources=self.time_after_source)

            mp.all_wait()
            if mp.am_master():
                self.q_results = []
                for mode in harminv_instance.modes:
                    self.q_results.append([mode.freq, mode.decay, mode.Q, abs(mode.amp)])

                self.q_results = np.array(self.q_results)

                if len(self.q_results.shape) == 1:
                    self.q_results = np.array([[self.fcen, 0, 0, 0]])

                self.q_results = self.q_results[self.q_results[:, 2] > 0]

                if self.q_results.shape[0] == 0:
                    self.q_results = np.array([[self.fcen, 0, 0, 0]])

                self.q_results = self.q_results[abs(self.q_results[:, 0] - self.fcen) < (self.df / 2)]

                if self.q_results.shape[0] == 0:
                    self.q_results = np.array([[self.fcen, 0, 0, 0]])

                return self.q_results[:, 2]
        else:
            raise Exception('Cannot run null simulation')

    def get_mode_volume(self, time_after_source=200):
        if self._simulation is not None:
            self.time_after_source = time_after_source
            self._simulation.run(until_after_sources=self.time_after_source)
            computational_cell = self._simulation.fields.total_volume()
            modal_volume = self._simulation.fields.modal_volume_in_box(where=computational_cell)

            mp.all_wait()
            if mp.am_master():
                self.modal_volume_result = modal_volume
                return modal_volume

        else:
            raise Exception('Cannot run null simulation')

    def get_cavity_mode(self, fcen=None, res=20):
        if self._simulation is not None:
            er_data_list = []
            if fcen is None:
                fcen = self.fcen

            for i in range(res):
                self._simulation.run(until=1 / fcen / res)
                er_data_list.append(self._simulation.get_array(mp.Vector3(), mp.Vector3(2 * self.sr, 0, self.sz),
                                                               mp.Er)[::-1].real)

            return np.array(er_data_list)
