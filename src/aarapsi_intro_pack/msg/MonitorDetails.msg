# note: this structure assumes the publisher/s and subscriber/s have pre-existing knowledge
# about what reference data to use and have a copy of the reference data they can access already

Header header		# Header
int64 matchId		# Index of VPR match
int64 trueId		# Index ground truth thinks is best (-1 if no groundtruth)
int8 state		# int matching 0: unknown, 1: bad, 2: good
float64 mState		# Continuous monitor state estimate 
float64 prob		# Monitor probability estimate
bool mStateBin		# Binary monitor state estimate

## Note: Indices are with respect to the reference data where the first item is index 0.
