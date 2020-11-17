## Problem Statement
In a football match, match simulations are based on single-point positional tracking data of players and the ball. However, in such match data-sets, often times, we do not have the entire data (i.e. there are missing data points) and then we have to make the necessary additions ourselves. One such question is whether the ball is in play or not (active or dead ball). This so-called ball-state is used by us as an input for the automatic direction of football matches. The data is from  a TRACAB dataset (not included in the repository).

The aim is to set up a sequential model that predicts the ball state (Dead / Alive) for each frame based on the player and ball position data. For this a "delay" of 100 frames is built in. This means that the player and ball positions from all previous frames, and from the 100 following frames may be used as input data.

### Proposed solution
Used an LSTM network to train a classifier to classify the ball_state as alive 
or dead, given a data point. The data point (for now) only contains information 
regarding the ball but should include the player positional data as well.
**The idea was to present a solution**
- To capture the sequential nature of the data, we compute the difference between 
each frame i.e. the given data points.
- The network trains on the first few frames (101 frams, masking the first one i.e. 
100 frames) and predicts label for the first (masked) data point.
- We iterate over all the data points, masking the data point at current 
iteration. We predict that data point based on our network.
- Additionally, once the data point is predicted, it is added to the training data.
- We save the network weights after training the model in an h5 file. 
- We retrain the network after 100 frames (rather than at every data point)

_NOTE: The model has not been tuned_

#### Additionally, certain alternative approaches are discussed in the python notebook 'thoughts.ipynb'
