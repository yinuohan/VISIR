os.chdir(cache)

datadir = '/Users/yinuo/Desktop/VISIR_HD141569/'

outdir = '/Users/yinuo/Desktop/Parametric fit/'

wavs = ['B9.7', 'B10.7', 'B12.4', 'PAH1', 'PAH2']

wav = wavs[2]

writeout = True

fit_star = 1


## Parameters
zoomfactor = 1

distance = 110.6 # pc
pixel = 0.045 * zoomfactor # arcsec
scale = distance * pixel

# i = 52.5 ± 4.5◦ and PA = 85.4 ± 1◦ (Augereau et al. 1999)
inclination_fit = 50
height_fit = 0


## Read in image
# Read image files
target = fits.getdata(datadir + f'HD141569_{wav}.fits')
psf = fits.getdata(datadir + f'PSF_{wav}.fits')
subtracted = fits.getdata(datadir + f'HD141569_{wav}.psub.fits')

# Pick image to fit to
if fit_star:
    im = target.copy()
else:
    im = subtracted.copy()

# Rotate image
rotation_angle = 87
im = rotate(im, angle=rotation_angle, reshape=False)

# Centre image
out = gridsearch(im, yrange=[-2, 2], xrange=[-2, 2], delta=0.1, plotgrid=True, smoothgrid=5)
im = out[0]

# Zoom image
im = zoom(im, zoomfactor) / zoomfactor**2

# Cut out image
im = reshape_image(im, 100, 100)

# Rotate kernel
kernel = rotate(psf, angle=rotation_angle, reshape=False)

# Zoom kernel
kernel = zoom(kernel, zoomfactor) / zoomfactor**2

# Reshape kernel to odd dimensions then centre and cut out
kernel = reshape_image(kernel, 241, 241)
kernel = shift_to_fitted_centre(kernel)
kernel = reshape_image(kernel, 99, 99)

# Normalise kernel
kernel /= kernel.sum()

if 1:
    plot(im)
    plot(kernel)
    plot180(im)

print('Background noise:', np.std(im[60:80, 60:80]))


## Make up rings
"Generate rings used to make model images"
y, x = im.shape

r_bounds_make = np.arange(0, y//2+0.01, 1)
r = (r_bounds_make[:-1] + r_bounds_make[1:]) / 2

weights_make = np.ones(len(r_bounds_make) - 1)
heights_make = height_fit
inclination_make = inclination_fit

image_rings = MakeImage(r_bounds_make, weights_make, heights_make, inclination_make, dim=y, n_points_per_pixel=200, kernel=kernel, scale=scale, rapid=False, add_before_convolve=False, verbose=False)
rings = image_rings.rings_make

"Image of point source"
starim = np.zeros([y, x])
starim[y//2-1:y//2+1, x//2-1:x//2+1] = 1/4
starim = convolve(starim, kernel)


## Function to generate model
"Gaussian function"
def gaussian(x, A, mu, sigma):
    return A * np.exp( - (x - mu)**2 / (2 * sigma**2) )

"Plot radial profile"
if 0:
    plt.figure()
    weights_make = gaussian(r, 1, 25, 5)
    plt.plot(r, weights_make)
    plt.show()

"Make image of Gaussian ring"
def gaussian_model(A=1, mu=25, sigma=5):
    weights_make = gaussian(r, A, mu, sigma)
    model_image = weighted_sum(weights_make, rings)
    return model_image

def gaussian_point_model(A=1, mu=25, sigma=5, star=0):
    return gaussian_model(A, mu, sigma) + star * starim


## Log likelihood function
if fit_star:
    def log_likelihood(theta):
        A, mu, sigma, star, err = theta
        model = gaussian_point_model(A, mu, sigma, star)
        sigma2 = err ** 2
        return -0.5 * np.sum((im - model) ** 2 / sigma2 + np.log(sigma2))

    def log_prior(theta):
        A, mu, sigma, star, err = theta
        if 0 <= A and 0 <= mu and 0 <= sigma and 0 <= star and 0 < err:
            return 0.0
        return -np.inf

else:
    def log_likelihood(theta):
        A, mu, sigma, err = theta
        model = gaussian_model(A, mu, sigma)
        sigma2 = err ** 2
        return -0.5 * np.sum((im - model) ** 2 / sigma2 + np.log(sigma2))

    def log_prior(theta):
        A, mu, sigma, err = theta
        if 0 <= A and 0 <= mu and 0 <= sigma and 0 < err:
            return 0.0
        return -np.inf

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


## MCMC
import emcee

# if wav == 'B10.7':
#     A_start = 0.0003
#     mu_start = 8.0
#     sigma_start = 1.5
#     star_start = 0.17
#     noise_start = np.std(im[60:80, 60:80])
# else:
A_start = 1.2 * np.mean(im[50, 56:60])
mu_start = 7.0
sigma_start = 2.0
star_start = im[49, 49] * 60
noise_start = np.std(im[60:80, 60:80])

if fit_star:
    pos = [A_start, mu_start, sigma_start, star_start, noise_start] + [A_start/100, mu_start/100, sigma_start/100, star_start/100, noise_start/100] * np.random.randn(32, 5)
else:
    pos = [1e-3, 6, 2, noise_start] + [3e-5, 0.1, 0.1, noise_start/30] * np.random.randn(32, 4)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
sampler.run_mcmc(pos, 5000, progress=True)
#sampler.run_mcmc(pos, 500, progress=True)


## Plot MCMC chains
fig, axes = plt.subplots(ndim, figsize=(10, 10), sharex=True)
samples = sampler.get_chain()

if fit_star:
    labels = ["A", "mu", "sigma", "star", "err"]
else:
    labels = ["A", "mu", "sigma", "err"]

for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
if writeout:
    plt.savefig(outdir + f'{wav}_chains.png', dpi=300)
plt.show()

# Get aurocorrelation time
tau = sampler.get_autocorr_time()
print(tau)


## Throw away points
flat_samples = sampler.get_chain(discard=2000, thin=1, flat=True)
print(flat_samples.shape)
np.save(outdir + f'{wav}_chains.npy', flat_samples)

# Corner plot
import corner

fig = corner.corner(flat_samples, labels=labels)
if writeout:
    plt.savefig(outdir + f'{wav}_corner.png', dpi=300)
plt.show()


## Fitted result
from IPython.display import display, Math

writetext = ''

print('Wavelength:', wav)
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "{3} = {0:.3e}_{{-{1:.3e}}}^{{+{2:.3e}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(txt)
    writetext += txt + '\n'
writetext += f'background_noise = {noise_start:.3e}'

if writeout:
    with open(outdir + f'{wav}_results.txt', 'w') as f:
        f.write(writetext)


## Plot curves drawn from posterior distribution
inds = np.random.randint(len(flat_samples), size=100)

drawn_profiles = []

fig, ax = plt.subplots(2, 3, figsize=(12, 7))

r2 = np.arange(0, y//2+0.01, 0.01)

# Plot samples drawn from the posterior distribution
for ind in inds:
    sample = flat_samples[ind]
    if fit_star:
        A, mu, sigma, star, err = sample
        model_profile = gaussian(r2, A, mu, sigma)
    else:
        A, mu, sigma, err = sample
        model_profile = gaussian(r2, A, mu, sigma)

    ax[0, 0].plot(r2, model_profile, "C0", alpha=0.05)
    drawn_profiles.append(model_profile)
drawn_profiles = np.array(drawn_profiles)

# Median model
if fit_star:
    A_median, mu_median, sigma_median, star_median, err_median = np.median(flat_samples, axis=0)
    median_model_image = gaussian_point_model(A_median, mu_median, sigma_median, star_median)
    median_model_profile = gaussian(r2, A_median, mu_median, sigma_median)
else:
    A_median, mu_median, sigma_median, err_median = np.median(flat_samples, axis=0)
    median_model_image = gaussian_model(A_median, mu_median, sigma_median)
    median_model_profile = gaussian(r2, A_median, mu_median, sigma_median)

# Model image, image, residuals, self-subtraction and PSF
ax[1, 0].imshow(im, origin='lower')
ax[1, 1].imshow(median_model_image, origin='lower')
ax[1, 2].imshow(im - median_model_image, origin='lower')
ax[0, 1].imshow(kernel, origin='lower')
ax[0, 2].imshow(im - rotate180(im), origin='lower')

# 3 sigma envelope
lines = np.quantile(drawn_profiles, [0.16, 0.5, 0.84], axis=0)
ax[0, 0].plot(r2, lines[1], color='C0')
ax[0, 0].plot(r2, median_model_profile, ':', color='C1')
ax[0, 0].fill_between(r2, lines[0], lines[2], color='C0', alpha=0.5, lw=0)

# Axis labels
ax[0, 0].set_title('Fitted profile')
ax[0, 0].set_xlabel('Radius (pixel)')
ax[0, 0].set_ylabel('Radial profile (image units)')
ax[0, 0].set_xlim([0, 15])

ax[0, 1].set_title('PSF')
ax[0, 1].set_xlabel('Pixel')
ax[0, 1].set_ylabel('Pixel')

ax[0, 2].set_title('Rotationally subtracted observation')
ax[0, 2].set_xlabel('Pixel')
ax[0, 2].set_ylabel('Pixel')
ax[0, 2].set_xlim([x//2 - window//2, x//2 + window//2])
ax[0, 2].set_ylim([y//2 - window//2, y//2 + window//2])

window = 50

ax[1, 0].set_title('Observation')
ax[1, 0].set_xlabel('Pixel')
ax[1, 0].set_ylabel('Pixel')
ax[1, 0].set_xlim([x//2 - window//2, x//2 + window//2])
ax[1, 0].set_ylim([y//2 - window//2, y//2 + window//2])

ax[1, 1].set_title('Model')
ax[1, 1].set_xlabel('Pixel')
ax[1, 1].set_ylabel('Pixel')
ax[1, 1].set_xlim([x//2 - window//2, x//2 + window//2])
ax[1, 1].set_ylim([y//2 - window//2, y//2 + window//2])

ax[1, 2].set_title('Observation - model')
ax[1, 2].set_xlabel('Pixel')
ax[1, 2].set_ylabel('Pixel')
ax[1, 2].set_xlim([x//2 - window//2, x//2 + window//2])
ax[1, 2].set_ylim([y//2 - window//2, y//2 + window//2])

plt.suptitle(wav)
plt.tight_layout()
plt.show()

if writeout:
    plt.savefig(outdir + f'{wav}_model.png', dpi=300)
