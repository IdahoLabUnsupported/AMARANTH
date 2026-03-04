# AMARANTH Open Source Grid Modeling
The AMARANTH testing framework provides an evaluation framework to test high technology readiness level (TRL) models prior to deployment in the electric sector. Through working with vendors and industry partners, AMARANTH determines the most impactful AI models to test to align with the energy sector’s goals. AMARANTH’s unique focus on model attack vectors and vulnerabilities gives the framework a unique perspective outside of the traditional data security space. The models are then fit into a testbed to analyze the effects of normal AI operation. Using faster than real-time simulations, AMARANTH can determine resilience and capacity through calculating model drift and amount of time it takes for a model to complete a task under normal and adversarial conditions. Finally, a report using quantitative metrics achieved in the previous step is shared with the vendor/industry partner to help inform the resilience of the model to cyberattacks, the capacity of the model in a realistic environment, and the amount of time a model can be kept in deployment before needing to be retrained. 

### Software Disclosure for Resilience Measurements
In this repository, we develop the AMARANTH resilience measurement methodology and test it out on three different open source projects that are relatively close to the postulated AI use case within the grid sector. We include resilience measurements pre-adversarial action and post-adversarial action with resilience measurement helpers contained [here](https://github.com/IdahoLabUnsupported/AMARANTH/tree/main/ResilienceMeasurementFramework). [AI Resilience Framework](https://hpcgitlab.hpc.inl.gov/lambpc/opensourcegridmodeling/-/blob/main/ResilienceMeasurementFramework/3.1.2_Measuring_Resilience_py.pdf?ref_type=heads) showcases how resilience was measured. [Attack Methodology and Framework](https://github.com/IdahoLabUnsupported/AMARANTH/blob/main/ResilienceMeasurementFramework/3.1.3_Attack_Methodology_and_Framework_1_np.pdf) details the types of attacks conducted and measurement methodology. The following code bases exhibit open-source uses to this dataset and results across multiple files, architectures, and use cases.

## Electricity Demand Austin TX [Electricity Demand Forecasting](https://github.com/IdahoLabUnsupported/AMARANTH/tree/main/ElectricityDemandAustinTX)
From the original authors: "Accurate forecasts for electricity demand are important for electric utilities and regional transmission organizations. Long term forecasts can reduce investment risk, medium term forecasts help in planning fuel purchases and scheduling plant maintenance, and short term forecasts are essential in matching electricity generation and demand for grid reliability.

This project focuses on medium term forecasting, specifically making one-week-out forecasts at a resolution of one hour for Austin, Texas. Historical electricity demand and weather data from 2002-2017 is used. A combination of timeseries analysis and regression models are used to make forecasting predictions, which are compared against the observed hourly electricity demand for a series of one-week intervals."

In AMARANTH, we desconstruct this ipynb notebook and use the model to inference data between 2017 to 2024, determining when drift occurs through developed resilience means for multiple [models](https://github.com/IdahoLabUnsupported/AMARANTH/tree/main/ElectricityDemandAustinTX/LoadForecastingAttacks/NormalModels). We additionally introduce attacks against each of the four models to determine the impact of the attacks recorded [here](https://github.com/IdahoLabUnsupported/AMARANTH/tree/main/ElectricityDemandAustinTX/LoadForecastingAttacks/Attacks). 

## Smart Meter Load Disaggregation [Electricity Load Disaggregation](https://github.com/IdahoLabUnsupported/AMARANTH/tree/main/Electricity-Load-Disaggregation)
From the original authors: "The smart meter usage has been increasing in the past years as people get more motivated to improve their energy management, reduce consumption and utility costs. But even though smart meters help us understand our aggregate power consumption, we are still poor at estimating how much energy individual appliances are using. This can lead to suboptimal choices when deciding what appliance to turn off or replace. This is when smart meter disaggregation (also known as Non-intrusive load monitoring or NILM) can be useful. To put simply smart meter disaggreggation is a process of extracting individual appliance power signatures from the total aggregate signal. It can be argued that directly measuring individual appliance consumption by means of installing smart plugs would be easier and lead to more precise measurements. While this is conceptually true, installing smart plugs is an expensive and cumbersome process and considering how many appliances an average household uses this solution will simply not scale."

The AMARANTH team used this repository to determine the effects of adversarial action and the resilience of classification and disaggregation tasks for grid applications, determining when drift occurs through developed resilience means in [model-testing.ipynb](https://github.com/IdahoLabUnsupported/AMARANTH/blob/main/Electricity-Load-Disaggregation/python_notebooks/model-testing.ipynb). We additionally introduce attacks against each of the four models to determine the impact of the attacks recorded [here](https://github.com/IdahoLabUnsupported/AMARANTH/tree/main/Electricity-Load-Disaggregation/python_notebooks). 

## Sponsor
This work is supported by Department of Energy (DOE), under DOE Idaho Operations Office Contract DE-AC07-05ID14517. Accordingly, the U.S. Government retains a nonexclusive, royalty-free license to publish or reproduce the published form of this contribution, or allow others to do so, for U.S. Government purposes. Artificial Intelligence Management and Research for Advanced Networked Testhub (AMARANTH) program sponsored by DOE Grid Deployment Office and performed by the Idaho National Laboratory as a part of contract DE-AC07-05ID14517. This research made use of Idaho National Laboratory’s High Performance Computing systems located at the Collaborative Computing Center and supported by the Office of Nuclear Energy of the U.S. Department of Energy and the Nuclear Science User Facilities under Contract No. DE-AC07-05ID14517.

## Data Sources
Attribution is proved to the following three data sources without which the analysis could not occur.
1. UKERC - [UK-DALE](https://ukerc.rl.ac.uk/cgi-bin/dataDiscover.pl?Action=detail&dataid=7d78f943-f9fe-413b-af52-1816f9d968b0)
2. NWS - [Historical Weather for Dallas Fort Worth Area](https://www.weather.gov/wrh/Climate?wfo=fwd)
3. ERCOT - [Load Forecasting](https://www.ercot.com/gridinfo/load/forecast)

## More Information

Resilience measurement methodology is contained in [AI Resilience Framework](https://github.com/IdahoLabUnsupported/AMARANTH/blob/main/ResilienceMeasurementFramework/3.1.2_Measuring_Resilience_py.pdf) and red teaming methodology is contained in [Attack Methodology and Framework](https://github.com/IdahoLabUnsupported/AMARANTH/blob/main/ResilienceMeasurementFramework/3.1.3_Attack_Methodology_and_Framework_1_np.pdf).

## Authors 
Patience Yockey Idaho National Laboratory  
Bradley Marx Idaho National Laboratory  
Lauren Ortiz Idaho National Laboratory  
Mckenzy Heavlin University of Illinois Urbana-Champaign  
Matthew "Ross" Kunz Idaho National Laboratory  
Alex Kaforey Idaho National Laboratory  
Max Taylor Boise State University  
Jeremy Jones Idaho National Laboratory  
Juliana Ocampo Giraldo Idaho National Laboratory  
