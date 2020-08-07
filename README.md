# GESM Model Implementation

This is a python implementation of the supply/demand equilibrium model, as well as a wrapper that allows optimization to be performed on total costs. Some examples can be seen [here](task-3-model.ipynb)

## Overall optimization

The transportation system model is wrapped in a framework that allows it to be used as the objective function for an arbitrary optimization model. Current variables that can be optimized over include

* Length of different subnetworks, specifically the amount of ROW dedicated to individual modes (e.g. bus lanes).
* Transit headways
* (This will be updated)

The overall optimization framework consists of several main objects:

* **Optimizer**: The optimizer is an implementation of any optimization software. It accepts a set of bounds and constraints and an objective function. It evaluates model inputs within a set of constraints until it reaches an "optimal" solution, defined by the result of the objective function.
* **Model container**: This translates the control variable returned by the optimizer into specific inputs and modifications to the transportation system model. It then tells the system model to run, gathers the outputs, and then combines them into an objective function result for the optimizer.
* **Model**: This, based on modifications from the model container, finds transportation system equilibrium and returns results to the model container.

![optimization](/images/Optimization.vpd.png)

## Model implementation

The model itself consists of three primary components that come into equilibrium with each other. They are:

* **Microtypes**: This is where the supply side of the model lives--it keeps track of the accumulation and speeds on every subnetwork in every microtype. It gets updated based on the mode choice decisions people make.
* **Choice characteristics**: This is where the information that informs mode choice decisions lives. This includes data such as the total travel time and waiting time for each mode over each possible trip that can be taken in the system. It gets updated based on the state of each subnetwork in each microtype.
* **Mode choice**: This is where the mode choices travelers make live, which depends on the utility functions associated with their user type and trip type. It gets updated based on the collected choice characteristics of the different modal options.

![model](/images/FHWA-sequence.vpd.png)

## Supply model

Within the supply model is where the inter-mode interactions and traffic system performance all interact. It has three main components:

* **Network collection**: This keeps track of all of the subnetworks within a microtype and assigns the accumulation of each vehicle type in each subnetwork.
* **Mode**: Each mode in each microtype is represented by a distinct object. These objects keep track of mode-specific information, such as headways, total accumulation, and commercial speeds.
* **Subnetwork**: Each subnetwork is also represented by a distinct object. Given the accumulation of each vehicle type (determined by the network collection object and the mode objects), this object updates its base travel speed using a subnetwork-specific quasi-MFD.

![model](/images/Supply-model.png)

