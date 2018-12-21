import os
import datetime
import numpy as np
import meep as mp

# refractive indices
n_Si = 3.7
n_SiN = 2.01
n_SiO2 = 1.46
n_TiO2 = 2.35
n_ITO = 1.85

# refractive index shortcuts
n_high = n_SiN
n_low = n_SiO2
n_bulk = n_Si
n_electrode = n_ITO


def system_command(command):
    if mp.am_master():
        os.system(command)

def create_directory(self):
    system_command('mkdir -p ' + self.directory)

def gaussian_beam(x, y, x0=0, theta=0, w0=0.5, n=1, l=0.785):
    return (np.exp((np.arctan(((x - x0)*l*np.sin(theta))/(n*np.pi*w0**2))*(-(n*np.pi*w0**2) +
            complex(0,1)*(x - x0)*l*np.sin(theta)) + (n*np.pi*(2*n*np.pi*w0**2*(x -
            x0)*np.sin(theta) - complex(0,1)*l*(y**2 + (x - x0)**2*np.cos(theta)**2 +
            2*(x - x0)**2*np.sin(theta)**2)))/l)/(complex(0,1)*n*np.pi*w0**2 + (x -
            x0)*l*np.sin(theta)))*np.pi)/np.sqrt(np.pi**2 + ((x -
            x0)**2*l**2*np.sin(theta)**2)/(n**2*w0**4))



class ChannelWaveguide:

    def __init__(self, n_ac, wg_width, wg_height, wg_gap, channel_width, channel_height, channel_overetch, substrate_thickness,
                 cladding_thickness):

        now = datetime.datetime.now()
        self.directory = 'log/' + now.strftime("%Y-%m-%d") + '/'

        self.n_ac = n_ac
        self.wg_width = wg_width
        self.wg_height = wg_height
        self.wg_gap = wg_gap
        self.channel_width = channel_width
        self.channel_height = channel_height
        self.channel_overetch = channel_overetch
        self.substrate_thickness = substrate_thickness
        self.cladding_thickness = cladding_thickness

        self._cell = None
        self._geometry = None
        self._sources = None
        self._simulation = None

    def define_cell(self, **kwargs):
        sx = kwargs.get('sx', None)
        if sx is None:
            self.sx = 20
        else:
            self.sx = sx
        sy = kwargs.get('sy', None)
        if sy is None:
            self.sy = self.wg_width * 5
        else:
            self.sy = sy
        sz = kwargs.get('sz', None)
        if sz is None:
            self.sz = self.wg_height * 10
        else:
            self.sz = sz
        dpml = kwargs.get('dpml', None)
        if dpml is None:
            self.dpml = 0.5
        else:
            self.dpml = dpml

        self._cell = mp.Vector3(self.sx + 2 * self.dpml, self.sy + 2 * self.dpml, self.sz + 2 * self.dpml)

    def define_geometry(self, norm=False):
        if self._cell is not None:
            self.norm = norm

            # define geometry
            self._geometry = []

            if norm:
                # add bulk
                bulk_thickness = (self.sx + 2 * self.dpml) / 2
                bulk_centre = - bulk_thickness / 2
                self._geometry.append(mp.Block(center=mp.Vector3(0, 0, bulk_centre),
                                               size=mp.Vector3(1e20, 1e20, bulk_thickness),
                                               material=mp.Medium(index=n_bulk)))

                # add substrate
                substrate_thickness = self.substrate_thickness + self.wg_height / 2
                substrate_centre = - substrate_thickness / 2
                self._geometry.append(mp.Block(center=mp.Vector3(0, 0, substrate_centre),
                                               size=mp.Vector3(1e20, 1e20, substrate_thickness),
                                               material=mp.Medium(index=n_low)))

                # add cladding
                cladding_thickness = self.cladding_thickness + self.wg_height / 2
                cladding_centre = cladding_thickness / 2
                self._geometry.append(mp.Block(center=mp.Vector3(0, 0, cladding_centre),
                                               size=mp.Vector3(1e20, 1e20, cladding_thickness),
                                               material=mp.Medium(index=n_low)))

                # add waveguide
                self._geometry.append(mp.Block(center=mp.Vector3(0, 0, 0),
                                               size=mp.Vector3(1e20, self.wg_width, self.wg_height),
                                               material=mp.Medium(index=n_high)))

            else:
                # add bulk
                bulk_thickness = (self.sx + 2 * self.dpml) / 2
                bulk_centre = - bulk_thickness / 2
                self._geometry.append(mp.Block(center=mp.Vector3(0, 0, bulk_centre),
                                               size=mp.Vector3(1e20, 1e20, bulk_thickness),
                                               material=mp.Medium(index=n_bulk)))

                # add substrate
                substrate_thickness = self.substrate_thickness + self.wg_height / 2
                substrate_centre = - substrate_thickness / 2
                self._geometry.append(mp.Block(center=mp.Vector3(0, 0, substrate_centre),
                                               size=mp.Vector3(1e20, 1e20, substrate_thickness),
                                               material=mp.Medium(index=n_low)))

                # add cladding
                cladding_thickness = self.cladding_thickness + self.wg_height / 2
                cladding_centre = cladding_thickness / 2
                self._geometry.append(mp.Block(center=mp.Vector3(0, 0, cladding_centre),
                                               size=mp.Vector3(1e20, 1e20, cladding_thickness),
                                               material=mp.Medium(index=n_low)))

                # add channel
                channel_centre = - self.wg_height / 2 - self.channel_overetch + self.channel_height / 2
                self._geometry.append(mp.Block(center=mp.Vector3(0, 0, channel_centre),
                                               size=mp.Vector3(self.channel_width, 1e20, self.channel_height),
                                               material=mp.Medium(index=self.n_ac)))

                # add waveguide
                self._geometry.append(mp.Block(center=mp.Vector3(0, 0, 0),
                                               size=mp.Vector3(1e20, self.wg_width, self.wg_height),
                                               material=mp.Medium(index=n_high)))

                # add waveguide rail
                self._geometry.append(mp.Block(center=mp.Vector3(0, 0,
                                                                 - self.wg_height / 2 - (self.channel_overetch+10) / 2),
                                               size=mp.Vector3(1e20, self.wg_width, self.channel_overetch+10),
                                               material=mp.Medium(index=n_low)))

                # add gap
                gap_centre = - self.wg_height / 2 - self.channel_overetch + (self.wg_height + self.channel_overetch) / 2
                self._geometry.append(mp.Block(center=mp.Vector3(0, 0, gap_centre),
                                               size=mp.Vector3(self.wg_gap, 1e20,
                                                               self.wg_height + self.channel_overetch),
                                               material=mp.Medium(index=self.n_ac)))

        else:
            raise Exception('Need a cell before defining geometry')

    def define_point_source(self, fcen=1.27389, df=0.1, x=0, y=0, z=0):
        if self._geometry is not None:
            self.fcen = fcen
            self.df = df
            self._sources = [mp.Source(src=mp.GaussianSource(self.fcen, fwidth=self.df),
                                       component=mp.Ey,
                                       center=mp.Vector3(x, y, z))]
        else:
            raise Exception('Need a geometry before defining sources')

    def define_transmission_source(self, fcen=1.27389, df=0.1, distance_from_pml=0.25):
        if self._geometry is not None:
            self.fcen = fcen
            self.df = df
            self._sources = [mp.Source(src=mp.GaussianSource(self.fcen, fwidth=self.df),
                                       component=mp.Ey,
                                       center=mp.Vector3(- self.sx / 2 + distance_from_pml, 0, 0))]
        else:
            raise Exception('Need a geometry before defining sources')

    def define_simulation(self, res=32):
        if (self._cell is not None) & (self._geometry is not None) & (self._sources is not None):
            self.res = res

            pml_layers = [mp.PML(self.dpml)]
            symmetries = [mp.Mirror(mp.X), mp.Mirror(mp.Y, phase=-1)]

            if self.norm:
                symmetries.append(mp.Mirror(mp.Z))
                self._simulation = mp.Simulation(cell_size=self._cell,
                                                 geometry=self._geometry,
                                                 boundary_layers=pml_layers,
                                                 resolution=res,
                                                 sources=self._sources,
                                                 default_material=mp.Medium(index=self.n_ac),
                                                 progress_interval=60,
                                                 symmetries = symmetries)
            else:
                self._simulation = mp.Simulation(cell_size=self._cell,
                                                 geometry=self._geometry,
                                                 boundary_layers=pml_layers,
                                                 resolution=res,
                                                 sources=self._sources,
                                                 progress_interval=60,
                                                 symmetries=symmetries)
        else:
            raise Exception('Environment not fully defined. Need a cell, geometry and sources')

    def define_transmission_simulation(self, res=32):
        if (self._cell is not None) & (self._geometry is not None) & (self._sources is not None):
            self.res = res

            pml_layers = [mp.PML(self.dpml)]
            symmetries = [mp.Mirror(mp.Y, phase=-1)]

            self._simulation = mp.Simulation(cell_size=self._cell,
                                             geometry=self._geometry,
                                             boundary_layers=pml_layers,
                                             resolution=res,
                                             sources=self._sources,
                                             progress_interval=60,
                                             symmetries=symmetries)
        else:
            raise Exception('Environment not fully defined. Need a cell, geometry and sources')

    def define_flux_regions(self, nfreq=100, distance_from_pml=0.5):
        if self._simulation is not None:
            self.nfreq = nfreq

            # back_power = mp.FluxRegion(center=mp.Vector3(- self.sx / 2 + distance_from_pml, 0, 0),
            #                            size=mp.Vector3(0, self.sy - 2 * distance_from_pml, self.sz - 2 * distance_from_pml),
            #                            direction=mp.X)
            # self.back_power = self._simulation.add_flux(self.fcen, self.df, nfreq, back_power)

            front_power = mp.FluxRegion(center=mp.Vector3(self.sx / 2 - distance_from_pml, 0, 0),
                                        size=mp.Vector3(0, self.sy - 2 * distance_from_pml, self.sz - 2 * distance_from_pml),
                                        direction=mp.X)
            self.front_power = self._simulation.add_flux(self.fcen, self.df, nfreq, front_power)

            # left_power = mp.FluxRegion(center=mp.Vector3(0, - self.sy / 2 + distance_from_pml, 0),
            #                            size=mp.Vector3(self.sx - 2 * distance_from_pml, 0, self.sz - 2 * distance_from_pml),
            #                            direction=mp.Y)
            # self.left_power = self._simulation.add_flux(self.fcen, self.df, nfreq, left_power)

            right_power = mp.FluxRegion(center=mp.Vector3(0, self.sy / 2 - distance_from_pml, 0),
                                        size=mp.Vector3(self.sx - 2 * distance_from_pml, 0, self.sz - 2 * distance_from_pml),
                                        direction=mp.Y)
            self.right_power = self._simulation.add_flux(self.fcen, self.df, nfreq, right_power)

            bottom_power = mp.FluxRegion(center=mp.Vector3(0, 0, - self.sz / 2 + distance_from_pml),
                                         size=mp.Vector3(self.sx - 2 * distance_from_pml, self.sy - 2 * distance_from_pml, 0),
                                         direction=mp.Z)
            self.bottom_power = self._simulation.add_flux(self.fcen, self.df, nfreq, bottom_power)

            top_power = mp.FluxRegion(center=mp.Vector3(0, 0, self.sz / 2 - distance_from_pml),
                                      size=mp.Vector3(self.sx - 2 * distance_from_pml, self.sy - 2 * distance_from_pml, 0),
                                      direction=mp.Z)
            self.top_power = self._simulation.add_flux(self.fcen, self.df, nfreq, top_power)

        else:
            raise Exception('Cannot add flux region to null simulation')

    def define_transmission_flux_regions(self, nfreq=100, distance_from_pml=0.5):
        if self._simulation is not None:
            self.nfreq = nfreq

            refl_flux = mp.FluxRegion(center=mp.Vector3(- self.sx / 2 + distance_from_pml, 0, 0),
                                       size=mp.Vector3(0, self.sy - 2 * distance_from_pml, self.sz - 2 * distance_from_pml),
                                       direction=mp.X)
            self.refl_flux = self._simulation.add_flux(self.fcen, self.df, nfreq, refl_flux)

            trans_flux = mp.FluxRegion(center=mp.Vector3(self.sx / 2 - distance_from_pml, 0, 0),
                                        size=mp.Vector3(0, self.sy - 2 * distance_from_pml, self.sz - 2 * distance_from_pml),
                                        direction=mp.X)
            self.trans_flux = self._simulation.add_flux(self.fcen, self.df, nfreq, trans_flux)

        else:
            raise Exception('Cannot add flux region to null simulation')

    def end_simulation(self):
        if self._simulation is not None:
            self._simulation.reset_meep()
            self._cell = None
            self._geometry = None
            self._sources = None
            self._simulation = None
        else:
            raise Exception('Cannot end a null simulation')

    def get_epsilon(self, res=32):
        if self._simulation is not None:
            self.define_simulation(res=res)
            self._simulation.init_fields()
            eps_data = self._simulation.get_array(center=mp.Vector3(), size=self._cell,
                                                  component=mp.Dielectric)[::-1]
            self.end_simulation()
            return eps_data
        else:
            raise Exception('Cannot get eps from null simulation')

    def get_beta(self, time_after_source=200):
        if self._simulation is not None:
            self.time_after_sources = time_after_source

            self._simulation.run(until_after_sources=time_after_source)

            self.front_power_results = np.array(mp.get_fluxes(self.front_power))
            self.right_power_results = np.array(mp.get_fluxes(self.right_power))
            self.top_power_results = np.array(mp.get_fluxes(self.top_power))
            self.bottom_power_results = np.array(mp.get_fluxes(self.bottom_power))
            self.total_power_results = np.array(
                2 * self.front_power_results + 2 * self.right_power_results + self.top_power_results - self.bottom_power_results)
            beta = self.front_power_results / self.total_power_results

            flux_freqs = np.array(mp.get_flux_freqs(self.front_power))
            self.beta_results = np.transpose(np.array([flux_freqs, beta]))
        else:
            raise Exception('Cannot get beta from null simulation')

    def get_transmission_norm_spectra(self, time_after_source=200):
        if self._simulation is not None:
            self._simulation.run(until_after_sources=time_after_source)

            self.flux_freqs = np.array(mp.get_flux_freqs(self.refl_flux))
            self.norm_reflection = np.array(mp.get_fluxes(self.refl_flux))
            self.norm_transmission = np.array(mp.get_fluxes(self.trans_flux))
            self._simulation.save_flux('refl-flux', self.refl_flux)

    def get_transmission_spectra(self, time_after_source=200):
        if self._simulation is not None:
            self._simulation.load_minus_flux('refl-flux', self.refl_flux)
            self._simulation.run(until_after_sources=time_after_source)

            self.flux_reflection = np.array(mp.get_fluxes(self.refl_flux))
            self.flux_transmission = np.array(mp.get_fluxes(self.trans_flux))

    def get_transmission(self, fcen=1.27389, df=1, res=20, nfreq=100, time_after_source=200):
        self.time_after_sources = time_after_source

        self.define_cell(sx=20, sy=5*self.wg_width, sz=10*self.wg_height)
        self.define_geometry(norm=True)
        self.define_transmission_source(fcen=fcen, df=df)
        self.define_transmission_simulation(res=res)
        self.define_transmission_flux_regions(nfreq=nfreq)
        self.get_transmission_norm_spectra(time_after_source=time_after_source)
        self.end_simulation()

        self.define_cell(sx=20, sy=5*self.wg_width, sz=10*self.wg_height)
        self.define_geometry(norm=False)
        self.define_transmission_source(fcen=fcen, df=df)
        self.define_transmission_simulation(res=res)
        self.define_transmission_flux_regions(nfreq=nfreq)
        self.get_transmission_spectra(time_after_source=time_after_source)
        os.system('rm refl-flux.h5')
        self.end_simulation()

    def get_transmission_at_frequency(self, fcen=1.27389, df=1, res=20, time_after_source=200):
        self.get_transmission(fcen=fcen, df=df, res=res, nfreq=1, time_after_source=time_after_source)
        return self.flux_transmission[0] / self.norm_transmission[0]


class GratingDefectExcitation:

    def __init__(self, n_ac, dbr_periodicity, dbr_fill_factor, top_dbr_number, bottom_dbr_number, cavity_length,
                 defect_height, buffer_thickness, electrode_thickness, grating_etch, grating_period,
                 grating_fill_factor, grating_number, grating_distance):

        now = datetime.datetime.now()
        self.directory = 'log/' + now.strftime("%Y-%m-%d") + '/'

        self.n_ac = n_ac
        self.dbr_periodicity = dbr_periodicity
        self.dbr_fill_factor = dbr_fill_factor
        self.top_dbr_number = top_dbr_number
        self.bottom_dbr_number = bottom_dbr_number
        self.cavity_length = cavity_length
        self.defect_height = defect_height
        self.buffer_thickness = buffer_thickness
        self.electrode_thickness = electrode_thickness

        self.grating_etch = grating_etch
        self.grating_period = grating_period
        self.grating_fill_factor = grating_fill_factor
        self.grating_number = grating_number
        self.grating_distance = grating_distance

        self._cell = None
        self._geometry = None
        self._sources = None
        self._simulation = None

    def define_cell(self, **kwargs):
        sx = kwargs.get('sx', None)
        if sx is None:
            self.sx = 20
        else:
            self.sx = sx
        sz = kwargs.get('sz', None)
        if sz is None:
            self.sz = 10
        else:
            self.sz = sz
        dpml = kwargs.get('dpml', None)
        if dpml is None:
            self.dpml = 1
        else:
            self.dpml = dpml

        self._cell = mp.Vector3(self.sx + 2 * self.dpml, 0, self.sz + 2 * self.dpml)

    def define_geometry(self, norm=False):
        if self._cell is not None:
            self.norm = norm

            # define geometry
            self._geometry = []

            if not norm:

                # from centre

                # cavity
                cavity_effective_height = self.cavity_length
                self._geometry.append(mp.Block(center=mp.Vector3(0, 0, 0),
                                               size=mp.Vector3(1e20, 1e20, cavity_effective_height),
                                               material=mp.Medium(index=self.n_ac)))
                current_layer_position = - cavity_effective_height / 2

                # Buffer layer
                current_layer_thickness = self.buffer_thickness
                self._geometry.append(
                    mp.Block(center=mp.Vector3(0, 0, current_layer_position - current_layer_thickness / 2),
                             size=mp.Vector3(1e20, 1e20, current_layer_thickness),
                             material=mp.Medium(index=n_low)))
                current_layer_position -= current_layer_thickness

                # Electrode layer
                current_layer_thickness = self.electrode_thickness
                self._geometry.append(
                    mp.Block(center=mp.Vector3(0, 0, current_layer_position - current_layer_thickness / 2),
                             size=mp.Vector3(1e20, 1e20, current_layer_thickness),
                             material=mp.Medium(index=n_electrode)))
                current_layer_position -= current_layer_thickness

                # Oxide / Nitride pairs
                for i in range(self.bottom_dbr_number):
                    current_layer_thickness = self.dbr_periodicity * (1 - self.dbr_fill_factor)
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(0, 0, current_layer_position - current_layer_thickness / 2),
                                 size=mp.Vector3(1e20, 1e20, current_layer_thickness),
                                 material=mp.Medium(index=n_low)))
                    current_layer_position -= current_layer_thickness

                    current_layer_thickness = self.dbr_periodicity * self.dbr_fill_factor
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(0, 0, current_layer_position - current_layer_thickness / 2),
                                 size=mp.Vector3(1e20, 1e20, current_layer_thickness),
                                 material=mp.Medium(index=n_high)))
                    current_layer_position -= current_layer_thickness

                # Top layers

                # from centre

                # Cavity in grating
                current_layer_position = - cavity_effective_height / 2
                current_grating_position = - self.grating_distance / 2

                for i in range(self.grating_number):
                    current_grating_thickness = self.grating_period * self.grating_fill_factor
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(current_grating_position - current_grating_thickness / 2, 0,
                                                   current_layer_position - self.grating_etch / 2),
                                 size=mp.Vector3(current_grating_thickness, 1e20, self.grating_etch),
                                 material=mp.Medium(index=self.n_ac)))
                    current_grating_position -= self.grating_period

                # Buffer layer
                current_layer_position = cavity_effective_height / 2
                current_grating_position = - self.grating_distance / 2

                for i in range(self.grating_number):
                    current_grating_thickness = self.grating_period * (1 - self.grating_fill_factor)
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(current_grating_position - current_grating_thickness / 2, 0,
                                                   current_layer_position - self.grating_etch / 2),
                                 size=mp.Vector3(current_grating_thickness, 1e20, self.grating_etch),
                                 material=mp.Medium(index=n_low)))
                    current_grating_position -= self.grating_period

                current_layer_thickness = self.buffer_thickness
                self._geometry.append(
                    mp.Block(center=mp.Vector3(0, 0, current_layer_position + current_layer_thickness / 2),
                             size=mp.Vector3(1e20, 1e20, current_layer_thickness),
                             material=mp.Medium(index=n_low)))
                current_layer_position += current_layer_thickness

                # Electrode layer
                current_grating_position = - self.grating_distance / 2

                for i in range(self.grating_number):
                    current_grating_thickness = self.grating_period * (1 - self.grating_fill_factor)
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(current_grating_position - current_grating_thickness / 2, 0,
                                                   current_layer_position - self.grating_etch / 2),
                                 size=mp.Vector3(current_grating_thickness, 1e20, self.grating_etch),
                                 material=mp.Medium(index=n_electrode)))
                    current_grating_position -= self.grating_period

                current_layer_thickness = self.electrode_thickness
                self._geometry.append(
                    mp.Block(center=mp.Vector3(0, 0, current_layer_position + current_layer_thickness / 2),
                             size=mp.Vector3(1e20, 1e20, current_layer_thickness),
                             material=mp.Medium(index=n_electrode)))
                current_layer_position += current_layer_thickness

                self.grating_position = current_layer_position

                current_grating_position = - self.grating_distance / 2

                for i in range(self.grating_number):
                    current_grating_thickness = self.grating_period * (1 - self.grating_fill_factor)
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(current_grating_position - current_grating_thickness / 2, 0,
                                                   current_layer_position - self.grating_etch / 2),
                                 size=mp.Vector3(current_grating_thickness, 1e20, self.grating_etch),
                                 material=mp.Medium(index=1)))
                    current_grating_position -= self.grating_period

                # Oxide / Nitride pairs
                self.dbr_block_width = 10
                current_dbr_position = self.grating_distance / 2 + self.dbr_block_width / 2

                for i in range(self.top_dbr_number):
                    current_layer_thickness = self.dbr_periodicity * (1 - self.dbr_fill_factor)
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(current_dbr_position, 0,
                                                   current_layer_position + current_layer_thickness / 2),
                                 size=mp.Vector3(self.dbr_block_width, 1e20, current_layer_thickness),
                                 material=mp.Medium(index=n_low)))
                    current_layer_position += current_layer_thickness

                    current_layer_thickness = self.dbr_periodicity * self.dbr_fill_factor
                    self._geometry.append(
                        mp.Block(center=mp.Vector3(current_dbr_position, 0,
                                                   current_layer_position + current_layer_thickness / 2),
                                 size=mp.Vector3(self.dbr_block_width, 1e20, current_layer_thickness),
                                 material=mp.Medium(index=n_high)))
                    current_layer_position += current_layer_thickness

            else:
                self.grating_position = 0

        else:
            raise Exception('Need a cell before defining geometry')

    def define_gaussian_source(self, fcen=1.27389, df=0.1, x0=0, theta=0, w0=0.5, distance_from_pml=0.1):
        if self._geometry is not None:
            self.fcen = fcen
            self.df = df
            self.source_x0 = x0
            self.source_theta = theta
            self.source_w0 = w0

            def source_amp_func(vec3=mp.Vector3()):
                emode = gaussian_beam(vec3.x, 0, x0=x0, theta=theta, w0=w0, n=1, l=1/fcen)
                return emode

            if self.norm:
                self._sources = [mp.Source(src=mp.GaussianSource(self.fcen, fwidth=self.df),
                                           component=mp.Ey,
                                           center=mp.Vector3(0, 0, 0),
                                           size=mp.Vector3(self.sx, 0, 0),
                                           amp_func=source_amp_func)]

            else:
                self._sources = [mp.Source(src=mp.GaussianSource(self.fcen, fwidth=self.df),
                                       component=mp.Ey,
                                       center=mp.Vector3(0, 0, self.sz / 2 - distance_from_pml),
                                       size=mp.Vector3(self.sx, 0, 0),
                                       amp_func=source_amp_func)]
        else:
            raise Exception('Need a geometry before defining sources')

    def define_simulation(self, res=32):
        if (self._cell is not None) & (self._geometry is not None) & (self._sources is not None):
            self.res = res

            pml_layers = [mp.PML(self.dpml)]

            if self.norm:
                self._simulation = mp.Simulation(cell_size=self._cell,
                                                 geometry=self._geometry,
                                                 boundary_layers=pml_layers,
                                                 resolution=res,
                                                 sources=self._sources,
                                                 default_material=mp.Medium(index=self.n_ac),
                                                 progress_interval=60)
            else:
                self._simulation = mp.Simulation(cell_size=self._cell,
                                                 geometry=self._geometry,
                                                 boundary_layers=pml_layers,
                                                 resolution=res,
                                                 sources=self._sources,
                                                 progress_interval=60)
        else:
            raise Exception('Environment not fully defined. Need a cell, geometry and sources')

    def define_flux_regions(self, nfreq=100, distance_from_pml=0.1):
        if self._simulation is not None:
            self.nfreq = nfreq

            if not self.norm:
                cavity_power = mp.FluxRegion(center=mp.Vector3(self.sx / 2 - distance_from_pml, 0, 0),
                                            size=mp.Vector3(0, 0, self.cavity_length),
                                            direction=mp.X)
                self.cavity_power = self._simulation.add_flux(self.fcen, self.df, nfreq, cavity_power)

                if (self.grating_distance / 2 + self.dbr_block_width) > (self.sx / 2):
                    top_flux_size = self.sx / 2 - self.grating_distance / 2
                    top_flux_position = self.grating_distance / 2 + top_flux_size / 2

                else:
                    top_flux_size = self.dbr_block_width
                    top_flux_position = self.grating_distance / 2 + top_flux_size / 2

                top_power = mp.FluxRegion(center=mp.Vector3(top_flux_position, 0, self.sz / 2 - distance_from_pml),
                                            size=mp.Vector3(top_flux_size, 0, 0),
                                            direction=mp.Z)
                self.top_power = self._simulation.add_flux(self.fcen, self.df, nfreq, top_power)

            else:
                norm_power = mp.FluxRegion(center=mp.Vector3(0, 0, self.sz / 2 - distance_from_pml),
                                            size=mp.Vector3(self.sx, 0, 0),
                                            direction=mp.Z)
                self.norm_power = self._simulation.add_flux(self.fcen, self.df, nfreq, norm_power)

        else:
            raise Exception('Cannot add flux region to null simulation')

    def end_simulation(self):
        if self._simulation is not None:
            self._simulation.reset_meep()
            self._cell = None
            self._geometry = None
            self._sources = None
            self._simulation = None
        else:
            raise Exception('Cannot end a null simulation')

    def get_epsilon(self):
        if self._simulation is not None:
            self._simulation.init_fields()
            eps_data = self._simulation.get_array(center=mp.Vector3(), size=self._cell,
                                                  component=mp.Dielectric)
            return np.transpose(eps_data)[::-1]
        else:
            raise Exception('Cannot get eps from null simulation')

    def get_movie(self, dt=1, number_images=100):
        if self._simulation is not None:
            ey_data_list = []

            for i in range(number_images):
                self._simulation.run(until=dt)
                ey_data_list.append(np.transpose(self._simulation.get_array(mp.Vector3(), self._cell,
                                                               mp.Ey).real)[::-1])

            return np.array(ey_data_list)
        else:
            raise Exception('Cannot get movie from null simulation')

    def calculate_source_power(self, time_after_source=100):
        if self._simulation is not None:

            self._simulation.run(until_after_sources=time_after_source)

            flux_freqs = np.array(mp.get_flux_freqs(self.norm_power))
            flux_norm = np.array(mp.get_fluxes(self.norm_power))
            self.norm_power_results = flux_norm
            self.norm_results = np.transpose(np.array([flux_freqs, flux_norm]))
        else:
            raise Exception('Cannot get source norm from null simulation')

    def calculate_grating_efficiency(self, time_after_source=100):
        if self._simulation is not None:

            self._simulation.run(until_after_sources=time_after_source)

            flux_freqs = np.array(mp.get_flux_freqs(self.cavity_power))
            flux_coupling = np.array(mp.get_fluxes(self.cavity_power))
            self.coupling_power_results = flux_coupling
            self.coupling_results = np.transpose(np.array([flux_freqs, flux_coupling/self.norm_power_results]))

            flux_up = np.array(mp.get_fluxes(self.top_power))
            self.top_power_results = flux_up
            self.leak_results = np.transpose(np.array([flux_freqs, flux_up/self.norm_power_results]))
        else:
            raise Exception('Cannot get source norm from null simulation')

    def get_transmission(self, sx=10, sz=5, x0=-3, theta=10/180*np.pi, distance_from_pml=0.1,
                         time_after_source=100):

        self.define_cell(sx=sx, sz=sz)
        self.define_geometry(norm=False)
        self.define_gaussian_source(fcen=self.fcen, df=self.df, x0=x0, theta=theta, w0=self.source_w0,
                                    distance_from_pml=distance_from_pml)
        self.define_simulation(res=self.res)
        self.define_flux_regions(nfreq=1, distance_from_pml=distance_from_pml)
        self.calculate_grating_efficiency(time_after_source=time_after_source)
        self.end_simulation()

        return (self.coupling_power_results[0]/self.norm_power_results[0],
                self.top_power_results[0]/self.norm_power_results[0])


class NanoBeamMirror:

    def __init__(self, wg_width=0.6891, wg_height=0.2741, period=0.2355, hole_x=0.1234, hole_y=0.3015):

        self.wg_width = wg_width
        self.wg_height = wg_height
        self.period = period
        self.hole_x = hole_x
        self.hole_y = hole_y

        self._cell = None
        self._geometry = None
        self._sources = None
        self._simulation = None

    def define_cell(self, **kwargs):
        sy = kwargs.get('sy', None)
        if sy is None:
            self.sy = 3 * self.wg_width
        else:
            self.sy = sy
        sz = kwargs.get('sz', None)
        if sz is None:
            self.sz = 3 * self.wg_height
        else:
            self.sz = sz
        dpml = kwargs.get('dpml', None)
        if dpml is None:
            self.dpml = max(self.sy, self.sz) / 10
        else:
            self.dpml = dpml

        self._cell = mp.Vector3(self.period, self.sy + 2 * self.dpml, self.sz + 2 * self.dpml)

    def define_geometry(self):
        if self._cell is not None:

            # define geometry
            self._geometry = []

            # add waveguide
            self._geometry.append(mp.Block(center=mp.Vector3(0, 0, 0),
                                           size=mp.Vector3(1e20, self.wg_width, self.wg_height),
                                           material=mp.Medium(index=n_high)))

            # add hole
            # self._geometry.append(mp.Ellipsoid(center=mp.Vector3(0, 0, 0),
            #                                    size=mp.Vector3(self.hole_x, self.hole_y, 1e20),
            #                                    material=mp.Medium(index=n_low)))

            self._geometry.append(mp.Ellipsoid(center=mp.Vector3(-self.period / 2, 0, 0),
                                               size=mp.Vector3(self.hole_x, self.hole_y, 1e20),
                                               material=mp.Medium(index=n_low)))

            self._geometry.append(mp.Ellipsoid(center=mp.Vector3(self.period / 2, 0, 0),
                                               size=mp.Vector3(self.hole_x, self.hole_y, 1e20),
                                               material=mp.Medium(index=n_low)))

        else:
            raise Exception('Need a cell before defining geometry')

    def define_point_source(self, fcen=1.27389, df=0.1, x_displacement=0):
        if self._geometry is not None:
            self.fcen = fcen
            self.df = df
            self.source_centre = mp.Vector3(x_displacement, 0, 0)
            self.source_coordinates = [self.source_centre.x + self.period / 2, self.source_centre.y + self.sy/2 + self.dpml,
                                       self.source_centre.z + self.sz/2 + self.dpml]
            self._sources = [mp.Source(src=mp.GaussianSource(self.fcen, fwidth=self.df),
                                       component=mp.Ey,
                                       center=self.source_centre)]
        else:
            raise Exception('Need a geometry before defining sources')

    def define_point_magnetic_source(self, fcen=1.27389, df=0.1, x_displacement=0):
        if self._geometry is not None:
            self.fcen = fcen
            self.df = df
            self.source_centre = mp.Vector3(x_displacement, 0, 0)
            self.source_coordinates = [self.source_centre.x + self.period / 2, self.source_centre.y + self.sy/2 + self.dpml,
                                       self.source_centre.z + self.sz/2 + self.dpml]
            self._sources = [mp.Source(src=mp.GaussianSource(self.fcen, fwidth=self.df),
                                       component=mp.Hz,
                                       center=self.source_centre)]
        else:
            raise Exception('Need a geometry before defining sources')

    def define_simulation(self, res=32):
        if (self._cell is not None) & (self._geometry is not None) & (self._sources is not None):
            self.res = res

            pml_layers = [mp.PML(self.dpml, direction=mp.Y),
                          mp.PML(self.dpml, direction=mp.Z)]

            symmetries = [mp.Mirror(mp.Y, phase=-1), mp.Mirror(mp.Z)]

            self._simulation = mp.Simulation(cell_size=self._cell,
                                             geometry=self._geometry,
                                             boundary_layers=pml_layers,
                                             resolution=res,
                                             sources=self._sources,
                                             symmetries=symmetries,
                                             default_material=mp.Medium(index=n_low),
                                             progress_interval=60)

        else:
            raise Exception('Environment not fully defined. Need a cell, geometry and sources')

    def get_epsilon(self):
        if self._simulation is not None:
            self._simulation.init_fields()
            eps_data = self._simulation.get_array(center=mp.Vector3(), size=self._cell,
                                                  component=mp.Dielectric)
            return eps_data
        else:
            raise Exception('Cannot get eps from null simulation')

    def end_simulation(self):
        if self._simulation is not None:
            self._simulation.reset_meep()
            self._cell = None
            self._geometry = None
            self._sources = None
            self._simulation = None
        else:
            raise Exception('Cannot end a null simulation')

    def get_dispersion(self, k_number=20, k_limits=None, time_after_sources=100):

        if self._simulation is not None:
            self.time_after_sources = time_after_sources

            if k_limits is None:
                self.k_points = mp.interpolate(k_number, [mp.Vector3(0), mp.Vector3(0.5 / self.period)])
            else:
                if k_limits[0] is None:
                    if k_limits[1] is None:
                        self.k_points = mp.interpolate(k_number, [mp.Vector3(0), mp.Vector3(0.5 / self.period)])
                    else:
                        self.k_points = mp.interpolate(k_number, [mp.Vector3(0), mp.Vector3(k_limits[1])])
                else:
                    if k_limits[1] is None:
                        self.k_points = mp.interpolate(k_number, [mp.Vector3(k_limits[0]),
                                                                  mp.Vector3(0.5 / self.period)])
                    else:
                        self.k_points = mp.interpolate(k_number, [mp.Vector3(k_limits[0]), mp.Vector3(k_limits[1])])

            self.freq_points = self._simulation.run_k_points(time_after_sources, self.k_points)

            return self.freq_points
        else:
            raise Exception('Cannot get eps from null simulation')

    def plot_dispersion(self, ax, delete_light_cone=True):

        for i in range(len(self.k_points)):

            k = self.k_points[i].x
            for freq in self.freq_points[i]:

                if delete_light_cone:
                    if freq.real < k / n_low:
                        ax.plot(k, freq.real, 'bo')
                else:
                    ax.plot(k, freq.real, 'bo')

    def get_mode(self, fcen=1.27389, df=0.01, kx=0, frames=20):
        self._simulation.reset_meep()
        self.define_point_source(fcen=fcen, df=df, x_displacement=self.source_centre.x)
        self.define_simulation(self.res)
        self._simulation.k_point = mp.Vector3(kx)

        self._simulation.run(until_after_sources=self.time_after_sources)

        ey_data_list = []
        for frame in range(frames):
            self._simulation.run(until=1 / fcen / frames)
            ey_data_list.append(self._simulation.get_array(mp.Vector3(), self._cell,
                                                           mp.Ey))
        return np.array(ey_data_list)