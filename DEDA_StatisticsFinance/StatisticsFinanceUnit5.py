"""
DEDA Unit 5
Basic Statistics and Visualisation in Python
Authors: Junjie Hu and Isabell Fetzer, 20220924 WKH 
"""


"""
Interactive Graphs
"""
"""
Plotly enables Python users to create beautiful interactive visualisations.
"""
# pip install yfinance
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import numpy as np

CURRENCY = 'EUR'
def getData(crypto):
  # Define time window
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    last_year_date = (now - timedelta(days=365)).strftime("%Y-%m-%d")
    start = pd.to_datetime(last_year_date)
    end = pd.to_datetime(current_date)
    data = yf.download(tickers=f'{crypto}-{CURRENCY}', start = start, end = end , interval = '1d')
    return data

# Call function to retrieve data on Bitcoin 
btc_data = getData('BTC')
# eth_data = getData('ETH')
# Plot graph
fig = go.Figure(
        data = [
            go.Candlestick(
                x = btc_data.index,
                open = btc_data.Open,
                high = btc_data.High,
                low = btc_data.Low,
                close = btc_data.Close
            ) 
        ] )
fig.update_layout(
    title = f'Time Series with Range slider for BTC',
    xaxis_title = 'Date',
    yaxis_title = f'Price ({CURRENCY})',
    xaxis_rangeslider_visible = True
)
fig.update_yaxes(tickprefix=CURRENCY)
fig.show()



"""
Generating numbers in numpy array
"""

# Giving the seed of random number generator
np.random.seed(1)
# Generate a 20*20 matrix with uniformly distributed random integers between 0 to 50
A = np.random.randint(low=0, high=50, size=(20, 20))
print(A)

"""
Matrix Operations
"""
"""
NumPy allows for many matrix operations.The inalg.inv() computes the (multiplicative) inverse of a matrix. The dot() method returns the dot product of two arrays. Identity() returns the identity array and eye() returns a  two-dimensional  array with ones on the diagonal and zeros elsewhere.
"""
# Note: A here is the random matrix from above
A_inv = np.linalg.inv(A) # Inverse matrix
print(A_inv)
dot_result = np.dot(A, A_inv) # Matrix multiplication operation
print(dot_result)
idn_matrix = np.identity(20) # Generate a 20*20 identity matrix
print(idn_matrix)
# Using .allclose() function to evaluate two matrices are equal within tolerance
np.allclose(dot_result, np.eye(20))  # True


"""
Eigenvalues
"""
"""
The transpose() function from Numpy can be used to calculate the transpose of a matrix. The linalg.eig() method computes the eigenvalues of a squared array while diag() extracts a diagonal or construct a diagonal array.
"""
A = [[3, 1],[1, 1]] # [[3, 1],[1, 1]] would create a 2x2 matrix
A_eig = np.linalg.eig(A)
# Now the Jordan decomposition A = Gamma*Lambda*Gamma^T
E_val = A_eig[0]
Gamma = A_eig[1]
Lambda = np.diag(E_val)
# Check the result, you might get something within numerical eps
AA = np.dot( np.dot(Gamma, Lambda), np.transpose(Gamma) )
print( np.allclose(AA, A) )  # True
# Calculation of the square root of A
Lambda12 = np.sqrt(Lambda)
A12 = np.dot( np.dot(Gamma, Lambda12), np.transpose(Gamma) )


"""
Fourier Transformation
"""
from PIL import Image
from numpy.fft import fft,ifft
import numpy as np
# Open the image by using Python Imaging Library(PIL)
image_before=Image.open('berlin_view.jpg')
# Decoding and encoding image to float number
image_int=np.fromstring(image_before.tobytes(), dtype=np.int8)
# Processing Fourier transform
fft_transformed=fft(image_int)
# Filter the lower frequency, i.e. employ a high pass
fft_transformed=np.where(np.absolute(fft_transformed) < 9e4,0,fft_transformed)
# Inverse Fourier transform
fft_transformed=ifft(fft_transformed)
# Keep the real part
fft_transformed=np.int8(np.real(fft_transformed))
# Output the image
image_output=Image.frombytes(image_before.mode, image_before.size, fft_transformed)
image_output.show()

                        
"""
Normal Distribution
"""
"""
To create an inert active plot for the probability and cumulative density function of the normal distribution we use the Plotly package. 
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import plotly.express as px
from ipywidgets import interact, interactive, fixed, interact_manual
# Set up figure size of plot 
plt.rcParams['figure.figsize'] = (10,8) # length and width 
plt.rcParams['figure.dpi'] = 120 # general box size

                        
"""
We define the general environment of our plot: size, colour and style. 
"""
# Plot the cdf and pdf in one figure
def f(mu,sigma, colour):
  fig = plt.figure(figsize=(10, 8))
  fig.subplots_adjust( hspace = 0.3)
  fig.patch.set_facecolor('#eeefef')
  plt.style.use('classic')

  # upper plot: pdf
  plt.subplot(2, 1, 1)  # (rows, columns, which one)
  x_axis = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
  plt.plot(x_axis, stats.norm.pdf(x_axis, mu, sigma), c= colour, linewidth= 2)
  plt.xlabel('X')
  plt.ylabel('pdf')
  plt.ylim(0,1)
  plt.xlim(-10,10)
  plt.title(f'Probability Density Function pdf for $\mu= {mu}$ and $\sigma= {round(sigma,2)}$', fontweight="bold")
  # lower plot: cdf
  plt.subplot(2, 1, 2)
  plt.plot(x_axis, stats.norm.cdf(x_axis, mu, sigma), c= colour, linewidth= 2)
  plt.xlabel('X')
  plt.ylabel('cdf') 
  plt.ylim(0,1)
  plt.xlim(-10,10)
  plt.title(f'Cumulative Density Function cdf for $\mu= {mu}$ and $\sigma= {round(sigma,2)}$', fontweight="bold")

colours = ['red', 'green', 'blue']
interact(f, mu=(-10, 10,1), sigma=(0.5, 5, 0.5), colour = colours)
plt.savefig("filename.png", transparent=True)

                        
"""
Kernel Density Estimation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
# Code reference: http://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html
# Set up figure size of plot 
plt.rcParams['figure.figsize'] = (10,8) # length and width 
plt.rcParams['figure.dpi'] = 120 # general box size
N = 200
np.random.seed(1)
# Create 2 normal distributed data set
norm_data_1 = np.random.normal(0, 1, int(0.3 * N))
norm_data_2 = np.random.normal(5, 1, int(0.7 * N))
norm_data = np.concatenate((norm_data_1, norm_data_2))
# Create x axis range
X_plot = np.linspace(-5, 10, 1000)
# Create linear combination of 2 normal distributed random variable
norm_linear = (0.3 * norm(0, 1).pdf(X_plot) + 0.7 * norm(5, 1).pdf(X_plot))
fig, ax = plt.subplots()
# Plot the real distribution
ax.fill(X_plot, norm_linear, fc='black', alpha=0.2, label='Linear combination')
# Use 3 different kernels to estimate
for kernel in ['gaussian', 'tophat', 'epanechnikov']:
    # Initial an object to use kernel function to fit data, bandwidth will    affect the result
    kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(norm_data.reshape(-1, 1))
    # Evaluate the density model on the data
    log_dens = kde.score_samples(X_plot.reshape(-1, 1))
    ax.plot(X_plot, np.exp(log_dens), '-', label="kernel = '{0}'".format(kernel))
# Add text on the plot, position argument can be arbitrary
ax.text(6, 0.38, "N={0} points".format(N))
# Add a legend to the left outside of the plot 
ax.legend(loc='upper left', bbox_to_anchor = (1,0.5))
# Plot the random points, squeeze them into narrow space
ax.plot(norm_data, -0.005 - 0.01 * np.random.random(norm_data.shape[0]), '+k')
# Set x-axis y-axis limit to adjust the figure
ax.set_xlim(-4, 9)
ax.set_ylim(-0.03, 0.4)
fig.savefig('kernel_estimation.png', dpi=300, transparent = True, bbox_inches='tight')
plt.show()

                        
"""
Box-Muller Method on 2-dim normal distribution
"""
"""
Using the Box-Muller Method to generate 2-dim normally distributed variables. 
"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(100) 
# For mu = (0,0), covariance matrix Sigma = identity matrix
n = 500  # Number of random numbers
msize = 0.1  # determines the size of the plotted points
# a good size might be msize=5 for n=500 pts and msize=0.1 for n>50K  
a = np.random.exponential(scale=1, size=n)
phi = np.random.uniform(low=0, high=2 * np.pi, size=n)
# Change to cartesian coordinates
x = a * np.cos(phi)
y = a * np.sin(phi)
plt.figure(figsize=(7, 7))
plt.plot(x, y,'.r', markersize=msize)
plt.savefig('2dimnormal.png', dpi = 300, transparent = True)

                        
"""
For covariance matrix Sigma = A: Y = X/sqrt(Sigma) ~ N(0,I) => Y*sqrt(Sigma)
"""
# Calculate sqrt(A) with Jordan decomposition
A = [[3, 1], [1, 1]]
A_eig = np.linalg.eig(A)
E_val = A_eig[0]
Gamma = A_eig[1]
Lambda = np.diag(E_val)
np.sqrt(Lambda)
Lambda12 = np.sqrt(Lambda)
A12 = np.dot(np.dot(Gamma, Lambda12), np.transpose(Gamma))
# Solve with matrix multiplication
c = [x, y]
tfxy = np.dot(A12, c)
plt.figure(2, figsize=(6, 4))
plt.plot(tfxy[0], tfxy[1], 'ro', markersize=msize)
