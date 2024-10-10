import marimo

__generated_with = "0.8.22"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    import torch
    return mo, np, plt, sns, torch


@app.cell
def __(mo):
    mo.md(
        """
        # Data generating process
        For this example, we generate the data from a mixture of Gaussians.
        You can modify the number of Gaussians, the relative probability masses of each, and the location ($\mu$) and scale ($\sigma^2$) parameters.
        """
    )
    return


@app.cell
def __(mo):
    dgp_n_gaussians_input = mo.ui.slider(start = 1, stop = 5, step =1, label = "Number of Gaussians", value = 2)
    return (dgp_n_gaussians_input,)


@app.cell
def __(dgp_n_gaussians_input, mo):
    mo.hstack([dgp_n_gaussians_input, mo.md(f"dgp_n: {dgp_n_gaussians_input.value}")])
    return


@app.cell
def __(dgp_n_gaussians_input, mo):
    dgp_n_gaussians = dgp_n_gaussians_input.value

    p_inputs_array = mo.ui.array( dgp_n_gaussians* [mo.ui.number(start = 0, stop = 1, value = 1/dgp_n_gaussians)] , label = "Probabilities of Gaussians")
    return dgp_n_gaussians, p_inputs_array


@app.cell
def __(mo):
    mo.md("""## Probability weights of Gaussians""")
    return


@app.cell
def __(dgp_n_gaussians, mo):
    mo.hstack(range(1, dgp_n_gaussians+1))
    return


@app.cell
def __(mo, p_inputs_array):
    mo.hstack(p_inputs_array)
    return


@app.cell
def __(mo, p_inputs_array):
    p_flag = True
    p_sum = sum(p_inputs_array.value)

    if p_sum != 1:
        p_flag = False

    ci= not p_flag
    mo.md("## Your probability weights do not add to one!") if ci else None
    return ci, p_flag, p_sum


@app.cell
def __(mo):
    mo.md("""## $\mu$""")
    return


@app.cell
def __(dgp_n_gaussians, mo):
    mu_inputs_array = mo.ui.array( [mo.ui.number(start = -10, stop = 10, value = 2*gaussian_idx - dgp_n_gaussians + 1) for gaussian_idx in range(dgp_n_gaussians)] , label = "Means of Gaussians")
    return (mu_inputs_array,)


@app.cell
def __(mo, mu_inputs_array):
    mo.hstack(mu_inputs_array)
    return


@app.cell
def __(mo):
    mo.md("""## $\sigma$""")
    return


@app.cell
def __(dgp_n_gaussians, mo):
    sigma_inputs_array = mo.ui.array( [mo.ui.number(start = 0, stop = 4, value = 1) for gaussian_idx in range(dgp_n_gaussians)] , label = "Means of Gaussians")
    return (sigma_inputs_array,)


@app.cell
def __(mo, sigma_inputs_array):
    mo.hstack(sigma_inputs_array)
    return


@app.cell
def __(np):
    def gmm_pdf(x, n, p_arr, mu_arr, sigma_arr):
        """
        Probability density function of a point in the mixture of Gaussians

        Parameters:
        - x: Tensor of point(s) at which pdf is to be evaluated
        - n: Number of Gaussian components
        - p_arr: A tensor of mixture probabilities for each Gaussian component.
        - mu_arr: A tensor of means for each Gaussian component.
        - sigma_arr: A tensor of standard deviations for each Gaussian component.

        Returns:
        - pdf: A tensor of sampled data points from the GMM.
        """
        pdf = 0
        for i in range(n):
            mu_i = mu_arr[i]
            sigma_i = sigma_arr[i]
            pdf += p_arr[i] * 1/np.sqrt(2 * np.pi * sigma_i**2) * np.exp(-(x-mu_i)**2/ 2 / sigma_i**2)

        return pdf
    return (gmm_pdf,)


@app.cell
def __(
    dgp_n_gaussians,
    gmm_pdf,
    mu_inputs_array,
    p_inputs_array,
    sigma_inputs_array,
    torch,
):
    x_range = torch.arange(-10, 10, 0.1)
    y = gmm_pdf(x_range, dgp_n_gaussians, p_inputs_array.value, mu_inputs_array.value, sigma_inputs_array.value)
    return x_range, y


@app.cell
def __():
    # plt.plot(x_range, y)
    return


@app.cell
def __(mo):
    mo.md(r"""## Samples""")
    return


@app.cell
def __(torch):
    def sample_gmm(n_samples, p_arr, mu_arr, sigma_arr):
        """
        Samples n data points from a mixture of Gaussians.

        Parameters:
        - n_samples: Number of samples to generate.
        - p_arr: A tensor of mixture probabilities for each Gaussian component.
        - mu_arr: A tensor of means for each Gaussian component.
        - sigma_arr: A tensor of standard deviations for each Gaussian component.

        Returns:
        - samples: A tensor of sampled data points.
        """
        num_components = len(p_arr)
        # print(f"num_components {num_components}")

        # Sample component indices from categorical distribution (based on p_arr)
        component_indices = torch.multinomial(p_arr, n_samples, replacement=True)

        samples = torch.zeros(n_samples)

        for component_idx in range(num_components):
            mask = (component_indices== component_idx)*1.
            component_samples = torch.normal(mu_arr[component_idx], sigma_arr[component_idx], size = (n_samples,))
            samples += mask* component_samples

        return samples
    return (sample_gmm,)


@app.cell
def __(mo):
    n_samples_input = mo.ui.number(start = 100, stop = 100000, value =10000, label = "Number of samples")
    n_samples_input
    return (n_samples_input,)


@app.cell
def __(
    mu_inputs_array,
    n_samples_input,
    p_inputs_array,
    sample_gmm,
    sigma_inputs_array,
    torch,
):
    n_samples = n_samples_input.value
    p_arr = torch.tensor(p_inputs_array.value)
    mu_arr = torch.tensor(mu_inputs_array.value)
    sigma_arr = torch.tensor(sigma_inputs_array.value)

    samples = sample_gmm(n_samples, p_arr, mu_arr, sigma_arr)
    return mu_arr, n_samples, p_arr, samples, sigma_arr


@app.cell
def __():
    return


@app.cell
def __(plt, sns):
    def plot_empirical_pdf(samples, bins=50):
        """
        Plots the empirical PDF and histogram counts of the given samples.

        Parameters:
        - samples: Tensor of data points sampled from the GMM.
        - bins: Number of bins to use for the histogram.
        - kde: Whether to overlay a Kernel Density Estimate (KDE) plot.
        """
        plt.figure(figsize=(8, 6))

        # Plot histogram with density=False so that we get counts
        sns.histplot(samples.numpy(), bins=bins, kde=True, stat="density", 
                     color="orange", edgecolor="white", label="Normalized histogram")

        # Plot KDE (if kde=True) separately

        sns.kdeplot(samples.numpy(), color="red", label="Empirical PDF (KDE)", linewidth=2)

        # Set labels and title
        plt.xlabel("Value", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.title("Histogram of Samples", fontsize=16)

        # Add legend
        plt.legend()

        return plt

    # Example usage (assuming `samples` contains the sampled data):
    # plot_empirical_pdf(samples)
    return (plot_empirical_pdf,)


@app.cell
def __(plot_empirical_pdf, samples, x_range, y):
    empirical_plt = plot_empirical_pdf(samples, bins=100)
    empirical_plt.plot(x_range, y, color = "green" ,label = "True PDF")
    empirical_plt.legend()
    empirical_plt.gca()
    return (empirical_plt,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # Simple Gaussian model with known variance

        We now assume that the samples we have are from a simple Gaussian distribution and we want to estimate its mean. We assume that we already know the variance of the Gaussian (so that we only have one parameter to deal with).

        This is an example of *model misspecification* since:
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Likelihood

        According to our model, all datapoints are i.i.d distributed as \(X_i \sim \mathcal N (  \theta, \lambda^2 ) \) where $\lambda^2$ is known.

        Thus, the likelihood of a single datapoint is:

        \[ f(X_i = x | \theta) = \frac{1}{\sqrt {2 \pi \lambda^2}} \exp \left[ -\frac{1}{2 \lambda^2} (x - \theta )^2 \right] \]

        And the likelihood of the whole data is:

        \[ f \left(X_1 = x_1, X_2 = x_2,... X_n = x_n | \theta \right) = \frac{1}{\sqrt { (2 \pi \lambda^2)^n}} \exp \left[ -\frac{1}{2 \lambda^2_0} \sum (x_i - \theta )^2 \right]  \]
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Prior

        We take a simple gaussian prior on the mean $\theta \sim \mathcal N ( \mu_0, \sigma_0^2 )$

        The probability density is thus:

        $$ \pi (\theta) = \frac{1}{\sqrt {2 \pi \sigma^2_0}} \exp \left[- \frac{1}{2 \sigma^2_0} (\theta - \mu_0)^2 \right] $$
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""### Specify the prior""")
    return


@app.cell
def __(mo):
    prior_mu_input = mo.ui.number(start = -10 , stop = 10, value = 0, label = "Prior mean")
    prior_var_input = mo.ui.number(start = 0, stop = 16, value = 1,  label = "Prior variance")
    return prior_mu_input, prior_var_input


@app.cell
def __(mo, prior_mu_input, prior_var_input):
    mo.hstack([prior_mu_input, prior_var_input])
    return


@app.cell
def __(mo):
    mo.md(r"""We can also make our assumption on $\lambda^2$ here.""")
    return


@app.cell
def __(mo):
    lambda_sq_input = mo.ui.number(start = 0 , stop = 16, value = 1, label = "$\lambda^2$")
    return (lambda_sq_input,)


@app.cell
def __(lambda_sq_input):
    lambda_sq_input
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Bayesian posterior

        By Bayes rule:

        \begin{align*}
        p(\theta | X_1 = x_1, X_2 = x_2,... X_n = x_n) &= \frac{ \pi (\theta) \cdot f(X_1 = x_1, X_2 = x_2,... X_n = x_n | \theta)}{\int \pi (\theta) \cdot f(X_1 = x_1, X_2 = x_2,... X_n = x_n | \theta) d\theta}\\
        &\propto \pi (\theta) \cdot f(X_1 = x_1, X_2 = x_2,... X_n = x_n | \theta)\\
        &\propto  \exp \left[-\frac{1}{2 \sigma^2_0} (\theta - \mu_0)^2- \frac{1}{2 \lambda^2} \sum_{i=1}^{n} ( \theta - x_i)^2 \right] 
        \end{align*}

        Simplifying, we get:

        $$ \theta | \mathbf X = \mathbf x \sim \mathcal N \left( \frac{\lambda^2 \mu_0 + \sigma_0^2 \sum_{i=1}^{n} x_i }{ \lambda^2+ n \sigma_0^2}, \frac{\lambda^2 \sigma^2_0}{ \lambda^2+ n \sigma_0^2}  \right) $$
        """
    )
    return


@app.cell
def __(torch):
    def posterior_pdf(x, data, lambda_sq, mu_0, sigma_0_sq):
        n = len(data)

        mu_posterior = (lambda_sq * mu_0  + sigma_0_sq * torch.sum(data))
        mu_posterior /= (lambda_sq + n* sigma_0_sq)

        sigma_sq_posterior = lambda_sq * sigma_0_sq
        sigma_sq_posterior /= (lambda_sq + n* sigma_0_sq)

        # print(mu_posterior)
        # print(sigma_sq_posterior)
        return torch.exp(torch.distributions.Normal(mu_posterior, torch.sqrt(torch.tensor(sigma_sq_posterior))).log_prob(x)), mu_posterior, sigma_sq_posterior
    return (posterior_pdf,)


@app.cell
def __(
    lambda_sq_input,
    posterior_pdf,
    prior_mu_input,
    prior_var_input,
    samples,
    x_range,
):
    y2, mu_post, sigma_sq_post = posterior_pdf(x_range, samples, lambda_sq_input.value, prior_mu_input.value, prior_var_input.value)
    return mu_post, sigma_sq_post, y2


@app.cell
def __(plt, x_range, y2):
    plt.plot(x_range, y2)
    return


@app.cell
def __(mo, mu_post, sigma_sq_post):
    mo.md(f"""Mean: {mu_post} \n
    Variance: {sigma_sq_post}""")
    return


@app.cell
def __():


    return


@app.cell
def __(mo):
    mo.md(r"""# Mixture of Gaussians model""")
    return


@app.cell
def __(mo):
    model_n_gaussians_input = mo.ui.slider(start = 1, stop = 5, step =1, label = "Number of Gaussians", value = 2)
    return (model_n_gaussians_input,)


@app.cell
def __(mo, model_n_gaussians_input):
    mo.hstack([model_n_gaussians_input, mo.md(f"model_n: {model_n_gaussians_input.value}")])
    return


@app.cell
def __(mo):
    mo.md(r"""## Prior""")
    return


@app.cell
def __(mo, model_n_gaussians_input):
    prior_mu_inputs_array = mo.ui.array( [mo.ui.number(start = -10, stop = 10, value = 2*gaussian_idx - model_n_gaussians_input.value + 1) for gaussian_idx in range(model_n_gaussians_input.value)] , label = "Prior on means of Gaussians")
    return (prior_mu_inputs_array,)


@app.cell
def __(mo, prior_mu_inputs_array):
    mo.hstack(prior_mu_inputs_array)
    return


app._unparsable_cell(
    r"""
    mu_prior_var = 
    """,
    name="__"
)


@app.cell
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
