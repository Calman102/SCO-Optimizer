# SCO Optimizer
 A genetic algorithm for multivariate optimization written in Python.
 
## Usage
Put `SplittingforContinuousOptimization.py` in your working directory and type
`from SplittingforContinuousOptimization import SCO`.

### Example
```python
from SplittingforContinuousOptimization import SCO, peaks

SCO(S=peaks, N=200, ς=0.8, w=0.5, B=[-3*np.ones(2), 3*np.ones(2)])
``` 

##References
1. Kroese, D. P., Botev, Z., Taimre, T., & Vaisman, R. (2019). *Data Science and Machine Learning : Mathematical and Statistical Methods*.
