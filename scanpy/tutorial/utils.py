from typing import Optional, Literal, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# install dask if available
try:
    import dask.array as da
except ImportError:
    da = None


plt.style.use("ggplot")


def toarray(matrix1d):
    return np.array(matrix1d).flatten()


setattr(np.matrix, 'toarray', toarray)


def read_mtx(path: str, gene_col: int = 0) -> Tuple[sparse.spmatrix, pd.DataFrame, pd.DataFrame]:
    genes = pd.read_csv(f'{path}/genes.tsv', sep='\t', usecols=[gene_col], header=None).set_index(gene_col)
    cells = pd.read_csv(f'{path}/barcodes.tsv', sep='\t', usecols=[0], header=None).set_index(0)

    non_zero = pd.read_csv(f'{path}/matrix.mtx', sep=' ', skiprows=2,
                            names=('gene_id', 'cell_id', 'counts'))
    shape = tuple(non_zero.iloc[0])[:2][::-1]
    non_zero.drop(0, inplace=True)
    non_zero.gene_id -= 1
    non_zero.cell_id -= 1

    adata = sparse.csc_matrix((non_zero.counts, (non_zero.cell_id, non_zero.gene_id)),
                              dtype=np.int16, shape=shape)

    return adata, genes, cells


def draw_four_plots(
    adata: sparse.spmatrix,
    genes: pd.DataFrame,
    cells: pd.DataFrame,
    capital: bool,
    min_gene_num: int,
    min_cell_num: int,
    max_gene_num: int,
    max_mt_pct: int
):
    # df = pd.DataFrame.sparse.from_spmatrix(adata, columns=genes.name)
    
    cell_filter = ((adata != 0).sum(axis=1) >= min_gene_num).toarray()
    gene_filter = ((adata != 0).sum(axis=0) >= min_cell_num).toarray()
    adata_filtered = adata[cell_filter][:, gene_filter]
    genes_filtered = genes[gene_filter].copy()
    cells_filtered = cells[cell_filter].copy()
    
    prefix = 'MT-' if capital else 'mt-'
    mt_filter = genes_filtered.index.str.startswith(prefix)
    
    gene_num_per_cell = (adata_filtered != 0).sum(axis=1).toarray()
    gene_counts_per_cell = adata_filtered.sum(axis=1).toarray()
    mt_pct = adata_filtered[:, mt_filter].sum(axis=1).toarray() / gene_counts_per_cell * 100
    
    g = sns.violinplot(data=gene_num_per_cell)
    g = sns.stripplot(data=gene_num_per_cell, jitter=0.4, size=2, color='.3')
    g.set_title('gene_num_per_cell')
    g.set_xticklabels(labels=[])
    plt.savefig('1.jpg')
    plt.cla()
    
    g = sns.violinplot(data=gene_counts_per_cell)
    g = sns.stripplot(data=gene_counts_per_cell, jitter=0.4, size=2, color='.3')
    g.set_title('gene_counts_per_cell')
    g.set_xticklabels(labels=[])
    plt.savefig('2.jpg')
    plt.cla()
    
    g = sns.violinplot(data=mt_pct)
    g = sns.stripplot(data=mt_pct, jitter=0.4, size=2, color='.3')
    g.set_title('pct_counts_mt')
    g.set_xticklabels(labels=[])
    plt.savefig('3.jpg')
    plt.cla()

    # cells_filtered['gene_counts_per_cell'] = gene_counts_per_cell
    # cells_filtered['mt_pct'] = mt_pct
    # cells_filtered['gene_num_per_cell'] = gene_num_per_cell
    cells_filtered = cells_filtered.assign(gene_counts_per_cell=gene_counts_per_cell, mt_pct=mt_pct, gene_num_per_cell=gene_num_per_cell)
    
    sns.scatterplot(data=cells_filtered, x='gene_counts_per_cell', y='mt_pct', s=7, alpha=0.5)
    plt.savefig('4.jpg')
    plt.cla()
    
    sns.scatterplot(data=cells_filtered, x='gene_counts_per_cell', y='gene_num_per_cell', s=7, alpha=0.5)
    plt.savefig('5.jpg')
    plt.cla()
    
    adata_filtered_filtered = adata_filtered[(gene_num_per_cell < max_gene_num) & (mt_pct < max_mt_pct)]
    
    return adata_filtered_filtered, genes_filtered, cells_filtered


def my_get_mean_var(X, axis: Union[Literal['gene', 'cell'], int]):
    if not isinstance(axis, int):
        axis = 0 if axis == 'gene' else 1

    if isinstance(X, sparse.spmatrix):  # same as sparse.issparse()
        X_copy = X.copy()
        X_copy.data **= 2
        mean = X.mean(axis=axis).toarray()
        var = X_copy.mean(axis=axis).toarray() - mean ** 2
        var *= X.shape[axis] / (X.shape[axis] - 1)
    else:
        mean = np.mean(X, axis=axis)
        var = np.var(X, axis=axis, ddof=1)  # a little overhead (mean counted twice, but it's ok.)
    return mean, var
'''
In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate of
the variance for normally distributed variables.
'''


def my_sparse_mean_variance_axis(mtx: sparse.spmatrix, axis: int):
    if isinstance(mtx, sparse.csr_matrix):
        ax_minor = 1
        shape = mtx.shape
    elif isinstance(mtx, sparse.csc_matrix):
        ax_minor = 0
        shape = mtx.shape[::-1]
    else:
        raise ValueError('This function only works on sparse csr and csc matrices')
    if axis == ax_minor:
        return my_sparse_mean_var_major_axis(
            mtx.data, mtx.indices, mtx.indptr, *shape, np.float64
        )
    else:
        return my_sparse_mean_var_minor_axis(
            mtx.data, mtx.indices, *shape, np.float64
        )
    

def my_sparse_mean_var_major_axis(
    data,
    indices,
    indptr,
    major_len,
    minor_len,
    dtype
):
    means = np.zeros(major_len, dtype=dtype)
    variances = np.zeros_like(means, dtype=dtype)  # why use zeros_like?
    for ind, (startptr, endptr) in enumerate(zip(indptr[:-1], indptr[1:])):
        counts = endptr - startptr
        
        mean = sum(data[startptr:endptr]) / minor_len
        variance = (sum((i-means[i]) ** 2 for i in data[startptr:endptr]) + mean ** 2 * (minor_len - counts)) / minor_len
        means[ind] = mean
        variances[ind] = variance
        
    return means, variances


def my_sparse_mean_var_minor_axis(
    data,
    indices,
    major_len,
    minor_len,
    dtype
):
    means = np.zeros(minor_len, dtype=dtype)
    variances = np.zeros_like(means, dtype=dtype)

    counts = np.zeros(minor_len, dtype=np.int64)
    
    for ind, num in zip(indices, data):
        means[ind] += num
    
    means /= major_len
    
    for ind, num in zip(indices, data):
        variances[ind] += (num - means[ind]) ** 2
        counts[ind] += 1
        
    variances += [mean ** 2 * (major_len - count) for mean, count in zip(means, counts)]
    variances /= major_len
    
    return means, variances


def highly_variable_genes_single_batch_seurat(
    adata: sparse.spmatrix,  # log transformed, base e
    genes: pd.DataFrame,
    layer=None,
    min_disp=0.5,
    max_disp=np.inf,
    min_mean=0.0125,
    max_mean=3,
    n_top_genes: int = 0,
    n_bins=20,
    flavor='seurat'
) -> None:
    X = adata.layers[layer] if layer is not None else adata#.X
    
    if flavor == 'seurat':
        # 如果不是以e为底的先变成以e为底
        X = np.expm1(X)
        # 然后还原
        
    mean, var = my_get_mean_var(X, axis='gene')
    mean[mean == 0] = 1e-12
    dispersion = var / mean
    if flavor == 'seurat':
        dispersion[dispersion == 0] = np.nan
        dispersion = np.log(dispersion)
        mean = np.log1p(mean)

    genes['dispersions'] = dispersion
    genes['means'] = mean
    genes['vars'] = var
    
    if flavor == 'seurat':
        genes['mean_bin'] = pd.cut(genes.means, bins=n_bins)
        disp_grouped = genes.groupby('mean_bin')['dispersions']
        
        single_bin_gene = []

        def find_nan_interval(x):
            if len(x) == 1:
                single_bin_gene.extend(x.index)
                std, mean = x.mean(), 0
            else:
                mean = x.mean()
                std = x.std(ddof=1)
            return (x - mean) / std
        
        genes['dispersions_norm'] = disp_grouped.transform(lambda x: find_nan_interval(x))
        if len(single_bin_gene) > 0:
            print(
                f'Gene indices {single_bin_gene} fell into a single bin: their '
                'normalized dispersion was set to 1.',
                '    Decreasing `n_bins` will likely avoid this effect.'
            )
    
    if n_top_genes > adata.shape[1]:
        print(f'`n_top_genes` > `adata.n_var`, returning all genes.')
        genes['highly_variable'] = np.ones(adata.shape[1], dtype=bool)
    elif n_top_genes > 0:
        genes_largest = genes.nlargest(n_top_genes, 'dispersion_norm')
        disp_cut_off = genes_largest.dispersion_norm[-1]
        genes['highly_variable'] = np.zeros(adata.shape[1], dtype=bool)
        genes.highly_variable.loc[genes_largest] == True
        print(
            f'the {n_top_genes} top genes correspond to a '
            f'normalized dispersion cutoff of {disp_cut_off}'
        )
    else:
        dispersion_norm = genes.dispersions_norm.values.astype('float32')
        np.nan_to_num(dispersion_norm)  # similar to Seurat
        gene_subset = np.logical_and.reduce(
            (
                mean > min_mean,
                mean < max_mean,
                dispersion_norm > min_disp,
                dispersion_norm < max_disp,
            )
        )
        genes['highly_variable'] = gene_subset

    sns.scatterplot(data=genes, x="means", y="dispersions", hue="highly_variable", s=7, alpha=0.5)
    plt.savefig('6.jpg')
    plt.cla()

    sns.scatterplot(data=genes, x="means", y="dispersions_norm", hue="highly_variable", s=7, alpha=0.5)
    plt.savefig('7.jpg')
    plt.cla()

    return None


if __name__ == '__main__':
    adata, genes, cells = read_mtx('data/hg19', gene_col=1)
    adata_filtered, genes_filtered, cells_filtered = draw_four_plots(
        adata, genes, cells, capital=True, min_gene_num=200, min_cell_num=3,
        max_gene_num=2500, max_mt_pct=5
        )

    adata_filtered_norm = normalize(adata_filtered, axis=1, norm='l1') * 1e4
    adata_filtered_norm_log1p = adata_filtered_norm.log1p()

    highly_variable_genes_single_batch_seurat(
        adata_filtered_norm_log1p, genes_filtered
    )
