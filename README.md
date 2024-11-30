# Ultrafast algorithm for synthesis of CRS spectra

## SYSTEM REQUIREMENTS

### Required software:
- Windows 10 or Ubuntu 20.02 or higher (MacOS is not currently supported)
- Python 3.8 or higher
- Python package dependencies: Numpy, Scipy, Matplotlib

### Required hardware:
- 64bit processor with AVX2 (i.e. about any processor less than ~10 years old)

### Tested versions:
- Windows 10 / Python 3.12
- Ubuntu 20.02 / Python 3.8



## INSTALLATION GUIDE

Installation will typically take a few minutes.

The example applications can be installed by executing the steps below. 
This procedure is written for Windows, we assume Linux users will be able to translate these to the equivalent Linux commands:

1. **Dowload repository**
   
    Using git, navigate to the folder where you want to clone the repo and type
    ```
    git clone https://github.com/dcmvdbekerom/ultrafast-crs
    ```
    Or:
   
    Navigate to https://github.com/dcmvdbekerom/ultrafast-crs using your webbrowser and download the zip file by clicking the green "<> Code v" button followed by "Download ZIP".
    Unzip “`ultrafast-crs-main.zip`” into a new folder (e.g. “`ultrafast-crs`”).

2. **Install Python**
   
    In case Python is not installed on your system, download the installer at https://www.python.org/ and follow the instructions to install Python. Make sure to check “Add Python 3.x to PATH” during installation. Reboot your system after installation. The applications were tested on Python 3.12 for Windows 10 and Python 3.8 on Ubuntu 20.04 (WSL2), but they will likely work with older versions of Python 3.x / Windows / Linux as well. MacOS is currently not supported.

3. **Install dependencies**
   
    Open a command prompt and navigate to the unzipped folder “ultrafast-crs”. You can do this quickly by navigating to the folder in Windows and typing “cmd” in the address bar and pressing enter. A console window should pop up with the current directory set to the “ultrafast-crs” folder. Now type:
    ```
    pip install -r requirements.txt
    ```
    in the console and press enter. This will install the Matplotlib, Numpy, and Scipy packages as well as their dependencies.

4. **Download database files**
   
    The database datafiles are stored on a separate repository (https://github.com/dcmvdbekerom/ultrafast-crs_data). 
    To download the data, run the following python script:
    ```
    python get_data.py
    ```
    The database files will be downloaded in the `data/CH4_v2/` folder. If anything went wrong, you can manually download the files from https://github.com/dcmvdbekerom/ultrafast-crs_data/data/CH4_v2/.

5. **Run widget application**
   
	With the console window still open, first navigate to the demos folder by typing:
	```
	cd demos
	```
	Now the time domain application can be run by typing:
    ```
    python demo_t-domain.py
    ```
    Or for the frequency domain application, type:
    ```
    python demo_w-domain.py
    ```


## DEMOS

###	Frequency-domain widget

The widget for the on-the-fly fitting of frequency-domain CRS spectra can be launched by running the Python script widgets_w.py. The script generates a graphical user interface (GUI) as shown in Fig. A2. The interactive plot consists of a synthetic CH4 ν2 CRS spectrum (circle markers) and the best-fit spectrum computed by the ufa-cpp algorithm (solid line).

The expected runtime of the fitting routine is <1s.

### Time-domain widget

This widget allows for on-the-fly evaluation of the time-domain measurement of the CRS signal, performed by scanning the probe delay, and can be launched by running the Python script widgets_t.py. The script generates the GUI shown in Fig. A3. The interactive plot consists of a synthetic CRS signal (circle markers) and the best-fit time-trace computed by the ufa-cpp algorithm (solid line).

The expected runtime of the fitting routine is <1s.



## INSTRUCTIONS FOR USE

###	Frequency-domain widget

![Frequency domain example](https://github.com/dcmvdbekerom/ultrafast-crs/blob/main/figures/output/widgets_w.png?raw=true)

1. **The target spectrum**
   
    Labelled as “data” in the legend, this synthetic spectrum is computed for a random temperature (T) in the range 296-1500 K and for a probe delay (τ) selected by the user in the range 20-100 ps, and simulates experimental data acquired in a typical fs/ps CRS experiment. The spectral resolution of this target spectrum is reduced to 2.0 cm-1, with respect to synthetic spectra shown in Fig. 3 in the main text. The synthetic data is normalized to unity. Noise can be added to the data by increasing the Noise slider to a non-zero value.

2. **The best-fit spectrum**
   
    Labelled as “fit” in the legend, this spectrum is calculated on-the-fly employing the ufa-cpp algorithm and including the full set of 10.9 M spectral lines of the MeCaSDa database1. It is worth reminding that, employing the ref-py algorithm, based on the state-of-the-art spectral code of Ref.2, a single evaluation of this spectrum requires many hours.

3. **Probe delay slider**
   
    This slider allows the user to select the probe delay of the data spectrum in the range 20-100 ps, the new value is shown on the right-hand side of the slider. When slid, a new data spectrum can be simulated for the new value of τ and for a new random value of T. When the slider is released, the “fit” spectrum is also recomputed for the new value of τ and the current value of T and scale.

4. **Noise slider**
   
    This slider allows the user to set the amount of noise that is added to the synthetic data. The set value determines the standard deviation of the Gaussian distribution from which the noise is sampled. It is expressed as a percentage of the maximum (normalized to 1).

5. **Temperature slider**
   
    This slider allows the user to change the temperature of the data spectrum in the range 296-1500 K, the new value is shown on the right-hand side of the slider. When slid, the fit spectrum is instantaneously recomputed for the new value of T and the current value of τ. The data spectrum is unaffected by changing the temperature slider, as the temperature of the spectrum is the unknown to be estimated, and its value is thus randomly selected by the widget.

6. **Scale slider**
   
    Allows the user to rescale the fit plot. The value (1 by default) is shown on the right-hand side of the slider. This scale is applied after the spectrum is normalized to 1, which is required because the data, which is also normalized to 1, may have its maximum at a different spectral location due to its much coarser sampling.

7. **New data**
   
    Clicking this button generates a new data spectrum at the current probe delay, and for a new randomly-selected temperature. It also recalculates the scale value that is required to perfectly reproduce the data. Both target values are indicated as red bars in the respective sliders for T and scale.

8. **Fit**
   
    When this button is clicked the data is fitted with a best-fit spectrum to obtain the current temperature. The best-fit spectrum is automatically updated.


### Time-domain widget

![Time domain example](https://github.com/dcmvdbekerom/ultrafast-crs/blob/main/figures/output/widgets_t.png?raw=true)

1. **The target trace**
   
    Labelled as “data” in the legend, this temporal trace corresponds to the value of a synthetic integrated Q-branch spectrum (amounting to 2.5 M lines) computed over 87 values of τ in the range 20-200 ps. The probe delay scan is here simulated for the input temperature value (in the range 296-1500 K), and for randomly-selected values of the MEG parameters α and β. Noise can be added to the data by increasing the Noise slider to a non-zero value.

2. **The best-fit trace**
   
    Labelled as “fit” in the legend, this temporal trace is calculated on-the-fly employing the ufa-cpp-t algorithm, and provides the fitted values of α and β.

3. **Temperature slider**
   
    This slider allows the user to change the temperature of the data trace in the range 296-1500 K, the new value is shown on the right-hand side of the slider. When sliding, the fit trace is recomputed live for the new value of T and the current values of α and β. When the slider is released, a new data trace is computed at current values for T, α and β.

4. **Noise slider**
   
    This slider allows the user to set the amount of noise that is added to the synthetic data. The set value determines the standard deviation of the Gaussian distribution from which the noise is sampled. It is expressed as a percentage of the local value of the data, reflecting the fact that during a probe delay scan the dynamic range of the digitizer (camera) can be readjusted at each probe delay.

5. **MEG parameter sliders**
   
    These two sliders allow the user to change two main parameters of the MEG scaling law that models the collisional dephasing of the Raman coherence. α can be varied in the range 0.002-0.1, β in the range 0-5, and the new value is shown on the right-hand side of the respective slider. The remaining MEG parameters are fixed at a = 2, δ = 1, and n = 0.

6. **New data**
   
    This button allows the user to generate a new data trace at the current temperatures, and for new randomly-selected values of the MEG parameters α and β. The selected values for α and β are indicated by the red lines on their respective sliders.

7. **Fit**
   
    This button allows the user to automatically fit the data trace to evaluate the corresponding MEG parameters. When clicked, the best-fitting trace is also updated.

