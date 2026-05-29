# prerequisite

- Install KNIME software version 5.8.3.
- Install python environment with python version 3.13.2. We recommend to install conda or Anaconda to facilitate the linkage between python environment and KNIME software.
- If you use Conda, install the following conda version: 25.5.1.
- When lauching KNIME software, select the downloaded folder "knime-workspace-git-GT-PhD" as the workspace.
- In KNIME software, in "Menu" > "Install Extensions", install the extension named "KNIME Python Integration".
- In the directory "Exported Workflows", import ".knwf" files into KNIME software. The imported workflows have to be located anywhere into the "knime-workspace-git-GT-PhD" folder.
- In KNIME software, in "Preferences" > "KNIME" > "Python", select your python environment between "Conda" or "Manual". If you use Conda, please indicate in "KNIME" > "Conda", the Conda installation directory.
- From your python environment, Install necessary librairies: snorkel, scikit-learn, umap-learn, rdp, scikit-learn-extra, lifelines, numpy, pandas, scipy, dask

There are 2 knwf files to import into KNIME:

1) data_preparation.knwf : Used to receive raw data and to prepare it
2) Configuration 1 : First configuration for patient clustering

You have to import them into knime in order to use them.
You have to execute them in order.
