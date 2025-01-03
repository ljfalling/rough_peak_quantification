import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    file_input  = mo.ui.file(kind='area', filetypes=[".vms"], multiple=False)
    file_input
    return file_input, mo


@app.cell
def _(file_input):
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".vms") as temp_file:
        temp_file.write(file_input.value[0].contents)
        filepath = temp_file.name
    return filepath, temp_file, tempfile


@app.cell
def _(file_input, filepath):
    from vamas import Vamas
    import numpy as np
    from dateutil import parser
    import pandas as pd

    # Access the selected file name and path:
    filename = file_input.value[0].name

    class AesStaib:
        """Class for handling files from Staib DESA

        Args:
            filepath (str): Path to .dat or .vms file
        """

        def __init__(self, filepath: str) -> None:
            self.ident = "AES"

            self.datetime = None
            self.mode = None
            self.dwell_time = None
            self.scan_num = None

            self.e_start = None
            self.e_stop = None
            self.stepwidth = None

            self.retrace_time = None
            self.res_mode = None
            self.res = None
            self.aes_data = None  # pyright: ignore[reportAttributeAccessIssue]
            self.script, self.div = None, None

            filename = filepath.split('/')[-1]

            if filename.endswith('.vms') and not filename.startswith('.'):
                self.read_staib_vamas(filepath)
            else:
                print('The file does not end with .vms or starts with . ')
                raise FileNotFoundError

        def read_staib_vamas(self, filepath: str) -> None:
            """Uses vamas library to read AES Staib .vms files

            Args:
                filepath (str): Path to the .vms file
            """

            vms = Vamas(filepath)
            data = vms.blocks[0] # AES Staib vms always contains only one block
            self.datetime = parser.parse(
                f"{data.year}-{data.month}-{data.day} {data.hour}:{data.minute}:{data.second}"
            )
            self.mode = data.signal_mode
            self.dwell_time = data.signal_collection_time
            self.scan_num = data.num_scans_to_compile_block

            self.e_start = data.x_start
            self.e_stop = (
                data.x_step * data.num_y_values + data.x_start
            ) - data.x_step
            self.stepwidth = data.x_step

            for i in data.additional_numerical_params:
                if i.label == "BKSrettime":
                    self.retrace_time = i.value
                elif i.label == "BKSresomode":
                    self.res_mode = i.value
                elif i.label == "BKSresol":
                    self.res = i.value
            x_values = np.linspace(
                data.x_start,
                (data.x_step * data.num_y_values + data.x_start) - data.x_step,
                num=data.num_y_values,
            )
            y_values = np.array(data.corresponding_variables[0].y_values)
            self.aes_data = np.column_stack((x_values, y_values))

    vms = AesStaib(filepath)
    aes_pd = pd.DataFrame(vms.aes_data, columns=['Kinetic energy / eV', 'Counts'])
    print('data dimensions: ', aes_pd.shape)
    return AesStaib, Vamas, aes_pd, filename, np, parser, pd, vms


@app.cell
def _(mo, np):
    whit_p = mo.ui.slider(steps=np.logspace(-5, -0.001, 50).tolist())
    whit_lam = mo.ui.slider(steps=np.logspace(-2, 8, 50).tolist())
    whit_diff = mo.ui.slider(steps=np.arange(2, 5).tolist())

    gaussian_filter_on = mo.ui.radio(options=['On', 'Off'], value='On', label='Apply Gaussian Filter:')
    sigma_slider = mo.ui.slider(steps=np.linspace(1, 30, 30).tolist())
    widths_range_slider = mo.ui.range_slider(steps=np.linspace(1, 30, 30).tolist())
    noise_perc_slider = mo.ui.slider(steps=np.linspace(1, 100, 100).tolist())
    return (
        gaussian_filter_on,
        noise_perc_slider,
        sigma_slider,
        whit_diff,
        whit_lam,
        whit_p,
        widths_range_slider,
    )


@app.cell
def _(
    gaussian_filter_on,
    mo,
    noise_perc_slider,
    sigma_slider,
    whit_diff,
    whit_lam,
    whit_p,
    widths_range_slider,
):
    mo.hstack((
        mo.vstack((
            mo.hstack([whit_p, mo.md(f"smoothing parameter p: {whit_p.value:.2e}")]),
            mo.hstack([whit_lam, mo.md(f"penalizing weighting factor: {whit_lam.value:.2e}")]),
            mo.hstack([whit_diff, mo.md(f"order of the differential matrix: {whit_diff.value}")])
        )),
        mo.vstack((
            mo.hstack([gaussian_filter_on, mo.md(f"Gaussian filter: {gaussian_filter_on.value}")]),
            mo.hstack([sigma_slider, mo.md(f"Gaussian sigma: {sigma_slider.value}")])
        )),
        mo.vstack((
            mo.hstack([widths_range_slider, mo.md(f"Peak finder kernel widths: {widths_range_slider.value[0]} - {widths_range_slider.value[1]}")]),
            mo.hstack([noise_perc_slider, mo.md(f"Peak finder noise percent: {noise_perc_slider.value}")])
        ))
    ))
    return


@app.cell
def _(
    aes_pd,
    gaussian_filter_on,
    mo,
    noise_perc_slider,
    np,
    sigma_slider,
    whit_diff,
    whit_lam,
    whit_p,
    widths_range_slider,
):
    # SavGol pre-smoothening did not help
    # first derivatives zero crossings are not useful either to find the shoulders, also not after SavGol smoothening
    # a gaussian_filter1d preprocessing enhances shoulder detection

    from scipy.signal import find_peaks_cwt
    from scipy.ndimage import gaussian_filter1d
    from plotly.subplots import make_subplots
    import pybaselines
    import plotly.graph_objects as go

    # Perform baseline correction using slider values
    aes_pd['Baseline'] = pybaselines.whittaker.derpsalsa(
        aes_pd['Counts'],
        p=whit_p.value,
        lam=whit_lam.value,
        diff_order=int(whit_diff.value)
    )[0]
    aes_pd['Background corrected counts'] = aes_pd['Counts'] - aes_pd['Baseline']

    if gaussian_filter_on.value == 'On':
        Y_blurred = gaussian_filter1d(aes_pd['Background corrected counts'], sigma=sigma_slider.value)
        Y_contrast = aes_pd['Background corrected counts'] - Y_blurred
    else:
        Y_contrast = aes_pd['Background corrected counts']

    # Define widths and noise_perc for peak finding
    widths = np.arange(widths_range_slider.value[0], widths_range_slider.value[1] + 1)
    noise_perc = noise_perc_slider.value

    # Peak finding
    rawpeaks = find_peaks_cwt(Y_contrast, widths, noise_perc=noise_perc)
    peak_heights = aes_pd['Background corrected counts'][rawpeaks]

    # Create the figure with two vertically stacked subplots
    fig1 = make_subplots(
        rows=1, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.03  # Adjust the spacing between plots
    )

    # Upper subplot: Original Counts and Baseline
    # Add the scatter plot of the original data
    fig1.add_trace(
        go.Scatter(
            x=aes_pd["Kinetic energy / eV"],
            y=aes_pd['Counts'],
            mode='markers',
            opacity=0.2,
            name='Counts',
            marker=dict(color='blue')
        ),
        row=1, col=1
    )

    # Add the baseline line plot
    fig1.add_trace(
        go.Scatter(
            x=aes_pd["Kinetic energy / eV"],
            y=aes_pd["Baseline"],
            mode='lines',
            name='Baseline',
            line=dict(color='orange')
        ),
        row=1, col=1
    )

    # Lower subplot: Background-Corrected Counts
    # Add the background-corrected data line plot
    fig1.add_trace(
        go.Scatter(
            x=aes_pd["Kinetic energy / eV"],
            y=aes_pd["Background corrected counts"],
            mode='lines',
            name='Background Corrected Counts',
            line=dict(color='red')
        ),
        row=1, col=2
    )

    fig1.add_trace(
        go.Scatter(
            x=aes_pd["Kinetic energy / eV"][rawpeaks],
            y=peak_heights,
            mode='markers',
            name='Peaks',
            marker=dict(color='black', symbol='circle', size=8)
        ),
        row=1, col=2
    )

    # Update x-axis and y-axis titles
    fig1.update_xaxes(title_text='Kinetic Energy (eV)', row=2, col=1)
    fig1.update_yaxes(title_text='Counts', row=1, col=1)
    fig1.update_yaxes(title_text='Background Corrected Counts', row=2, col=1)

    # Update layout and legend
    fig1.update_layout(
        title='AES Data with Baseline Correction',
        height=700,  # Adjust the figure height as needed
        width=1200,
        legend=dict(x=0.01, y=0.99),  # Position the legend
        showlegend=True
    )

    # Display the plot using marimo
    plot = mo.ui.plotly(fig1)
    mo.hstack([plot])
    return (
        Y_blurred,
        Y_contrast,
        fig1,
        find_peaks_cwt,
        gaussian_filter1d,
        go,
        make_subplots,
        noise_perc,
        peak_heights,
        plot,
        pybaselines,
        rawpeaks,
        widths,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
