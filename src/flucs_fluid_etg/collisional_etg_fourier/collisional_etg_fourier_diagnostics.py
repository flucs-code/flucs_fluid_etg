import cupy as cp
import numpy as np
from flucs.diagnostic import FlucsDiagnostic, FlucsDiagnosticVariable
#TODO remove these when coding optimisation wrapper
BLOCK_SIZE = int(256)
THREADS_PER_WARP = int(32) 

class HeatfluxDiag(FlucsDiagnostic):
    name = "heatflux"

    temp: cp.ndarray
    result: cp.ndarray
    heatflux_kx_kernel: cp.RawKernel
    last_axis_sum_nx_kernel: cp.RawKernel

    def init_vars(self):
        self.add_var(FlucsDiagnosticVariable(
            name="heatflux",
            shape=(),
            dimensions={},
            is_complex=False
        ))

    def ready(self):
        # Allocate temporary memory
        self.temp_zx = cp.zeros(self.system.nz * self.system.nx,
                                dtype=self.system.complex)

        self.temp_z = cp.zeros(self.system.nz, dtype=self.system.complex)
        self.result = cp.zeros((1,), dtype=self.system.complex)

        # Get kernels
        self.heatflux_kzkx_kernel = self.system.cupy_module.get_function("heatflux_kzkx")
        self.last_axis_sum_nx_kernel = self.system.cupy_module.get_function("last_axis_sum_nx")
        self.last_axis_sum_nz_kernel = self.system.cupy_module.get_function("last_axis_sum_nz")


    def execute(self):
        phi = self.system.phi[self.system.current_step % 2]
        T = self.system.T[self.system.current_step % 2]

        self.heatflux_kzkx_kernel(
                (self.system.nx * self.system.nz,),
                (BLOCK_SIZE,),
                (phi, T, self.temp_zx),
                shared_mem=THREADS_PER_WARP * self.system.complex().nbytes)

        self.last_axis_sum_nx_kernel(
                (self.system.nz,),
                (BLOCK_SIZE,),
                (self.temp_zx, self.temp_z),
                shared_mem=THREADS_PER_WARP * self.system.complex().nbytes)

        self.last_axis_sum_nz_kernel(
                (1,),
                (BLOCK_SIZE,),
                (self.temp_z, self.result),
                shared_mem=THREADS_PER_WARP * self.system.complex().nbytes)

        self.vars["heatflux"].data_cache.append(-1.5*self.result.item().real)


class FreeEnergyDiag(FlucsDiagnostic):
    name = "free_energy"

    temp: cp.ndarray
    result: cp.ndarray
    free_energy_kx_kernel: cp.RawKernel
    last_axis_sum_nx_kernel: cp.RawKernel

    def init_vars(self):
        self.add_var(FlucsDiagnosticVariable(
            name="W",
            shape=(),
            dimensions={},
            is_complex=False
        ))

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt",
            shape=(),
            dimensions={},
            is_complex=False
        ))

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_coll",
            shape=(),
            dimensions={},
            is_complex=False
        ))

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_inj",
            shape=(),
            dimensions={},
            is_complex=False
        ))

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_hyperdissipation_perp",
            shape=(),
            dimensions={},
            is_complex=False
            )
        )

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_hyperdissipation_kx",
            shape=(),
            dimensions={},
            is_complex=False
            )
        )

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_hyperdissipation_ky",
            shape=(),
            dimensions={},
            is_complex=False
            )
        )

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_hyperdissipation_kz",
            shape=(),
            dimensions={},
            is_complex=False
            )
        )

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_error",
            shape=(),
            dimensions={},
            is_complex=False
        ))


    def ready(self):
        # Allocate temporary memory
        self.temp_zx = cp.zeros(self.system.nz * self.system.nx,
                                dtype=self.system.complex)

        self.temp_z = cp.zeros(self.system.nz, dtype=self.system.complex)

        self.result = cp.zeros((1,), dtype=self.system.float)
        self.complex_result = cp.zeros((1,), dtype=self.system.complex)

        # Get kernels
        self.heatflux_kzkx_kernel = self.system.cupy_module.get_function("heatflux_kzkx")
        self.dW_kzkx_kernel = self.system.cupy_module.get_function("dW_kzkx")
        self.free_energy_kzkx_kernel = self.system.cupy_module.get_function("free_energy_kzkx")
        self.free_energy_collisional_loss_kzkx_kernel = self.system.cupy_module.get_function("free_energy_collisional_loss_kzkx")
        self.last_axis_sum_nx_kernel = self.system.cupy_module.get_function("last_axis_sum_nx")
        self.last_axis_sum_nz_kernel = self.system.cupy_module.get_function("last_axis_sum_nz")
        self.real_last_axis_sum_nx_kernel = self.system.cupy_module.get_function("real_last_axis_sum_nx")
        self.real_last_axis_sum_nz_kernel = self.system.cupy_module.get_function("real_last_axis_sum_nz")

        self.hyperdissipation_magnitude_kernels = {
            "perp": self.system.cupy_module.get_function("hyperdissipation_perp_magnitude"),
            "kx": self.system.cupy_module.get_function("hyperdissipation_kx_magnitude"),
            "ky": self.system.cupy_module.get_function("hyperdissipation_ky_magnitude"),
            "kz": self.system.cupy_module.get_function("hyperdissipation_kz_magnitude"),
        }

    def execute(self):
        # W
        fields = self.system.fields[self.system.current_step % 2]
        fields_prev = self.system.fields[(self.system.current_step - 1) % 2]

        self.free_energy_kzkx_kernel(
                (self.system.nx * self.system.nz,),
                (BLOCK_SIZE,),
                (fields, self.temp_zx),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

        self.real_last_axis_sum_nx_kernel(
                (self.system.nz,),
                (BLOCK_SIZE,),
                (self.temp_zx, self.temp_z),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

        self.real_last_axis_sum_nz_kernel(
                (1,),
                (BLOCK_SIZE,),
                (self.temp_z, self.result),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

        self.save_data("W", self.result.get().item())

        # dW/dt

        self.dW_kzkx_kernel(
                (self.system.nx * self.system.nz,),
                (BLOCK_SIZE,),
                (fields, fields_prev, self.temp_zx),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

        self.real_last_axis_sum_nx_kernel(
                (self.system.nz,),
                (BLOCK_SIZE,),
                (self.temp_zx, self.temp_z),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

        self.real_last_axis_sum_nz_kernel(
                (1,),
                (BLOCK_SIZE,),
                (self.temp_z, self.result),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

        dWdt = self.result.get().item() / self.system.current_dt
        self.save_data("dWdt", dWdt)

        # dW/dt_coll
        self.free_energy_collisional_loss_kzkx_kernel(
                (self.system.nx * self.system.nz,),
                (BLOCK_SIZE,),
                (fields, self.temp_zx),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

        self.real_last_axis_sum_nx_kernel(
                (self.system.nz,),
                (BLOCK_SIZE,),
                (self.temp_zx, self.temp_z),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

        self.real_last_axis_sum_nz_kernel(
                (1,),
                (BLOCK_SIZE,),
                (self.temp_z, self.result),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

        dWdt_coll =  -self.result.get().item()
        self.save_data("dWdt_coll", dWdt_coll)

        # dW/dt_inj
        phi = self.system.phi[self.system.current_step % 2]
        T = self.system.T[self.system.current_step % 2]

        self.heatflux_kzkx_kernel(
                (self.system.nx * self.system.nz,),
                (BLOCK_SIZE,),
                (phi, T, self.temp_zx),
                shared_mem=THREADS_PER_WARP * self.system.complex().nbytes)

        self.last_axis_sum_nx_kernel(
                (self.system.nz,),
                (BLOCK_SIZE,),
                (self.temp_zx, self.temp_z),
                shared_mem=THREADS_PER_WARP * self.system.complex().nbytes)

        self.last_axis_sum_nz_kernel(
                (1,),
                (BLOCK_SIZE,),
                (self.temp_z, self.complex_result),
                shared_mem=THREADS_PER_WARP * self.system.complex().nbytes)


        dWdt_inj = -self.system.input["parameters.kappaT"] * 1.5 * self.complex_result.get().item().real
        self.save_data("dWdt_inj", dWdt_inj)

        # Hyperdissipation

        # Free energy is a weighted sum of phi and T with the following weights
        charge = self.system.input["parameters.charge"]
        tratio = self.system.input["parameters.tratio"]
        taubar = tratio / charge

        # Factors of 2 account for the factor of 1/2
        # the one gets out of the partial_t derivatives
        # of quadratic quantities
        phi_weight = 2 * (1 + 1 / taubar) / (2 * taubar)
        T_weight = 2 * 0.75

        dWdt_hyperdissipation_total = 0.0
        for component, kernel in self.hyperdissipation_magnitude_kernels.items():
            dWdt_hyperdissipation_component = 0.0

            for field, weight in ((phi, phi_weight), (T, T_weight)):
                kernel(
                    (self.system.nx * self.system.nz,),
                    (BLOCK_SIZE,),
                    (field, self.temp_zx),
                    shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

                self.real_last_axis_sum_nx_kernel(
                    (self.system.nz,),
                    (BLOCK_SIZE,),
                    (self.temp_zx, self.temp_z),
                    shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

                self.real_last_axis_sum_nz_kernel(
                        (1,),
                        (BLOCK_SIZE,),
                        (self.temp_z, self.result),
                        shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

                dWdt_hyperdissipation_component += -self.result.get().item() * weight

            self.save_data(
                f"dWdt_hyperdissipation_{component}",
                dWdt_hyperdissipation_component
            )
            dWdt_hyperdissipation_total += dWdt_hyperdissipation_component


        self.save_data("dWdt_error", dWdt - dWdt_inj - dWdt_coll - dWdt_hyperdissipation_total)
