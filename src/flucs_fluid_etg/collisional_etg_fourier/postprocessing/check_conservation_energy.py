import argparse
import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt
from flucs.postprocessing import FlucsPostProcessing

def free_energy_check(post):

    # Get valid files for the specified variable
    nc_paths = post.get_valid_netcdf_paths("free_energy/dWdt")

    # Iterate over output files
    for index, nc_path in enumerate(nc_paths):

        # Separate figure for each output
        fig, axs = plt.subplots(3, 1, layout='constrained', sharex=True)
        ax_energy, ax_balance, ax_error = axs

        # Set figure title
        figure_name = f"check_conservation_energy_{pl.Path(nc_path).parent.name}"
        fig.canvas.manager.set_window_title(figure_name)

        # Read data from netCDF file
        variables = post.get_netcdf_variables(nc_path)

        time, boundaries, _ = post.load_netcdf_variable(nc_path, "time")
        dt = post.load_netcdf_variable(nc_path, "dt")[0]
        free_energy = post.load_netcdf_variable(nc_path, "free_energy/W")[0]
        dWdt = post.load_netcdf_variable(nc_path, "free_energy/dWdt")[0]
        injection = post.load_netcdf_variable(nc_path, "free_energy/dWdt_inj")[0]
        dissipation = post.load_netcdf_variable(nc_path, "free_energy/dWdt_coll")[0]

        # Add hyperdissipation
        for variable in variables:
            if variable.startswith("free_energy/dWdt_hyperdissipation_"):
                dissipation += post.load_netcdf_variable(nc_path, variable)[0]

        # Add vertical lines to mark restart boundaries
        for ax in axs:
            for index in boundaries:
                ax.axvline(time[index], color='black', linestyle="dotted")

        # Plot free energy
        ax_energy.plot(time, free_energy, label="W (free energy)", linewidth=1.5, color='black')

        # Plot free-energy balance
        ax_balance.plot(time, dWdt, label="dW/dt", linewidth=1.5, color='black', linestyle='solid')
        ax_balance.plot(time, injection, label="Injection", linewidth=1.5, color='red', linestyle='solid')
        ax_balance.plot(time, dissipation, label="Dissipation", linewidth=1.5, color='blue', linestyle='solid')
        ax_balance.plot(time, injection + dissipation, label="Injection + dissipation", linewidth=1.5, color='black', linestyle='dashed')

        # Plot error normalised to the timestep
        error = (dWdt - injection - dissipation)/dt
        ax_error.plot(time[1:], np.abs(error[1:]), label="Error / dt", linewidth=1.5, color='black')

        # Setting plot options
        ax_error.set_xlim(np.min(time), np.max(time))
        ax_error.set_xlabel(r"$(2c_s/L_B)t$")
        ax_error.set_yscale("log")

        ax_energy.legend()
        ax_balance.legend(ncols=2)
        ax_error.legend()

        # Save figures if required
        post.save(fig, name=figure_name, suffix="png", save_kwargs={"dpi": 300, "close": True})

        plt.show()


    return

if __name__ == "__main__":

    # Setup parser
    parser = argparse.ArgumentParser(
        parents=[FlucsPostProcessing.parser()], 
        description="Check the conservation laws of the Collisional ETG system.",
    )

    args = parser.parse_args()

    # Initialise post-processing object
    post = FlucsPostProcessing(
        io_paths=args.io_path,
        save_directory=args.save_directory,
        output_files=["output.0d.nc"],
        constraint="both"
    )

    # Call function
    free_energy_check(post)
