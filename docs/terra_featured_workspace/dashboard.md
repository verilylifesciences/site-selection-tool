The Baseline Site Selection Tool (BSST) is a decision support tool built for vaccine trial planners, especially for when models of future disease prevalence are unreliable. BSST is being developed for the public good with the initial goal of expediting clinical validation of vaccine candidates for COVID-19, with a focus on targeted site selection to support enhanced recruitment for vaccine research. 

BSST can quantify and visualize the likely outcome of specific vaccine trial plans, allows for interactive scenario evaluation, and implements mathematical optimizations to recommend alternatives. BSST operates down to the county/hospital level and can incorporate any epidemiological model(s) or predictions.

BSST allows the user to input model forecasts for regional disease prevalence, as well as historical disease incidence data. One potential source of US-only model forecasts is the [CDC ensemble](https://www.cdc.gov/coronavirus/2019-ncov/cases-updates/forecasts-cases.html). A potential source of historical data is [Google Cloud’s open repository](https://github.com/GoogleCloudPlatform/covid-19-open-data). For more detail, please see:
* [Baseline Site Selection Tool: Putting COVID models to work planning vaccine trials](https://github.com/verilylifesciences/site-selection-tool/blob/main/BaselineSiteSelectionTool.pdf) (PDF)
* https://github.com/verilylifesciences/site-selection-tool

![white space](https://storage.googleapis.com/terra-featured-workspaces/QuickStart/white-space.jpg)    

Different trial designs will have different additional requirements for the simulations. You can reach the Baseline Site Selection Tool team at site-selection-tool@projectbaseline.com  to discuss how those changes can be added to this tool.

![white space](https://storage.googleapis.com/terra-featured-workspaces/QuickStart/white-space.jpg)    

----------------------------
## How long will it take to run? How much will it cost?
**Time:** It takes about 2 minutes to run each notebook. Spend as much or as little time as you like using the interactive visualizations to explore the data.

**Cost:** Using the default Cloud Runtime configuration, the Terra notebook runtime charges are $0.20/hour for Google Cloud service costs. It should cost less than a quarter to run the notebooks.

![white space](https://storage.googleapis.com/terra-featured-workspaces/QuickStart/white-space.jpg)    

----------------------------
## Get Started

1. Clone this workspace.
1. Open `run_me_first.ipynb` and `Cell -> Run all`.
1. Then open any other notebook and  `Cell -> Run all`.

![white space](https://storage.googleapis.com/terra-featured-workspaces/QuickStart/white-space.jpg)    

----------------------------
## Notebooks

* **run_me_first** : This notebook will install the [Baseline Site Selection Tool](https://github.com/verilylifesciences/site-selection-tool) on your Terra  [detachable persistent disk](https://support.terra.bio/hc/en-us/articles/360047318551-Detachable-Persistent-Disks-).
* **trial_specification_demo** : Use the Baseline Site Selection Tool to specify and emit one or more different trial plans.
* **site_activation_demo** : For a particular trial plan, visualize how the trial is proceeding at a per site level and experiment with what will happen when you turn up or down different sites.

### Cloud Environment

Recommendation: Use the "default environment" which includes the following:

| Option | Value |
| --- | --- |
| Environment | Default: (GATK 4.1.4.1, Python 3.7.9, R 4.0.3) |
| CPU Minimum | 4|
| Disk size Minimum | 50 GB |
| Memory Minimum | 15 GB |

----------------------------
## Next steps

* Try these notebooks on your own data.
* Take a look at https://github.com/verilylifesciences/site-selection-tool to see the code and demonstration data used in this workspace.
* Ask questions by [filing an issue](https://github.com/verilylifesciences/site-selection-tool/issues) or emailing site-selection-tool@projectbaseline.com.

![white space](https://storage.googleapis.com/terra-featured-workspaces/QuickStart/white-space.jpg)    

----------------------------
### How to use these notebooks with your own data

When you want to **upload a CSV** to your workspace bucket, do this using the cloud console.

![white space](https://storage.googleapis.com/terra-featured-workspaces/QuickStart/white-space.jpg)    

![how to open the cloud console from Terra](https://storage.googleapis.com/terra-featured-workspaces/QuickStart/S51b_Workspaces_Google_bucket_Screen%20Shot.png)

![white space](https://storage.googleapis.com/terra-featured-workspaces/QuickStart/white-space.jpg)    

Now, from the Baseline Site Selection Tool notebooks, you can **refer to the file in the workspace bucket** like so:
```
os.path.join(os.environ['WORKSPACE_BUCKET'], 'path/to/the/file/you/uploaded.csv')
```

![white space](https://storage.googleapis.com/terra-featured-workspaces/QuickStart/white-space.jpg)    

----------------------------
# Workspace change log
Please see https://github.com/verilylifesciences/site-selection-tool/releases for the change log.
