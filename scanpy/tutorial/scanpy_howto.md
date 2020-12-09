## preprocessing

### highly_variable_genes

##### _highly_variable_genes_single_batch

```python
def _highly_variable_genes_single_batch(
    adata: AnnData,
    layer: Optional[str] = None,
    min_disp: Optional[float] = 0.5,
    max_disp: Optional[float] = np.inf,
    min_mean: Optional[float] = 0.0125,
    max_mean: Optional[float] = 3,
    n_top_genes: Optional[int] = None,
    n_bins: int = 20,
    flavor: Literal['seurat', 'cell_ranger'] = 'seurat',
) -> pd.DataFrame:
```



1. 是否提供`layer`
   - 若提供，从`adata`中筛选出相应数据
   - 否则用`adata`里的所有数据
2. `flavor`是否为`seurat`
   - 若是，如果数据做过log，就还原回去；没有做过log就直接`np.expm1`
   - 若不是原数据就不动
3. 对每个基因计算在所有细胞中表达量的均值和方差（无偏），均值为0的设置为很小的值
4. `dispersion = var / mean`
5. `flavor`是否为`seurat`
   - 若是，dispersion中的空值设置为`nan`，然后取log，均值取log1p
   - 若不是，什么也不做
6. `flavor`是否为`seurat`
   - 如果是
     1. 依据均值用`pd.cut`把表分成`n_bins`个组，每个组内计算均值和标准差（无偏）
     2. 那些只有一个基因的组的标准差会是`nan`，这些基因的标准差设置为均值，均值设置为0
     3. `dispersions_norm`是各组组内进行`dispersion`减均值除标准差
   - 如果是`cell_ranger`
     1. 不按`n_bins`分，而按分位数（10, 15, 20, 100）分成20个组，不计算均值和标准差，而计算组内中位数和绝对中位差`mad`（用`statsmodels.robust.mad`），这是对样本偏差的一种鲁棒性度量
     2. `dispersions_norm`也是改用中位数算
7. `dispersions_norm`转成`float32`得到`dispersion_norm`
8. 是否给定`n_top_genes`
   - 如果给定
     1. 挑出`dispersion_norm`最大的`n_top_genes`个基因就是`highly_variable`
   - 如果没给
     1. 找出`mean`在`min_mean`和`max_mean`之间的、`dispersion_norm`在`min_disp`和`max_disp`之间的就是`highly_variable`



### _simple

##### regress_out

Regress out (mostly) unwanted sources of variation.

```python
def regress_out(
    adata: AnnData,
    keys: Sequence[str],
    n_jobs: Optional[int] = None,
    copy: bool = False,
) -> Optional[AnnData]:
```

1. `_util.sanitize_anndata`

2. `keys[0]`在`adata.obs_keys`中且是分类变量吗

   - 如果是，那`keys`必须只有一个元素，否则报错。对`adata`中每个基因，按细胞类别求均值，用均值代替原值，得到`regressors`。`variable_is_categorical`设置为`True`
   - 如果不是
     1. `keys`为空吗
        - 如果`keys`为空，`regressors`设置为`adata.obs`
        - 否则`regressors`设置为`adata.obs`
     2. 把一个全为1的列插到`regressors`的第一列
     3. `variable_is_categorical`设置为`False`

3. 把`adata`按列分为`n_jobs`块

   实现细节：分`adata`的时候别直接分，如果`adata`的列数太少（小于1000），就把列数当成1000来分。用到`np.ceil`和`np.array_split`

4. `variable_is_categorical`为`True`吗

   - 如果是，那`regressors`形状和`adata`一样，把`regressors`也按列分为`n_jobs`块，分法也一样，每个任务的参数就是`regressors`和`adata`中对应的块
   - 如果不是，`regressors`不分块，每个任务的参数是`adata`中的块和`regressors`

5. 按`adata`的数量调用`multiprocessing.Pool().map_async`，在每个核上调用`_regress_out_chunk`

   1. 将每份结果竖着堆叠在一起，然后转置，按照参数`copy`是否为真，返回`adata`或`None`



##### _regress_out_chunk

```python
def _regress_out_chunk(data):
```

1. `data_chunk = data[0]`，`regressors = data[1]`，`variable_is_categorical = data[2]`

2. `responses_chunk_list = []`

3. 对`data_chunk`的每一列

   - 如果这一列都是同一个值，那就直接加入`responses_chunk_list`里面，因为这种情况没必要做regression，而且丢到`GLM`里也会报错

   - 如果不是

     1. `variable_is_categorial is True`吗？
        - 如果是，那用一个全为1的列和`regressors`相应的列拼成`regres`
        - 如果不是，那`regres`就直接是`regressors`
     2. `result = sm.GLM(data_chunk[:, col_index], regres, family=sm.families.Gaussian()).fit()`
     3. `new_column = result.resid_response`
     4. 出现`PerfectSeparationError`吗
        - 如果出现，`new_column = np.zeros(data_chunk.shape[0])`
        - 如果没有，什么也不做
     5. `new_column`加到`responses_chunk_list`里

4. 用`np.vstack`把`responses_chunk_list`并起来返回
   

##### scale

```python
def scale_array(
  	X,
  	*,
  	zero_center: bool = True,
  	max_value: float = -0.0,
  	copy: bool = False,
  	return_mean_std: bool = False
)
```

1. 如果`zero_center is False and max_value > 0`，be careful，也就是最好在zero_center的情况下使用max_value
2. 如果`X`是整型，需要转成浮点型
3. `mean, var = _get_mean_var(X); std = np.sqrt(var)`
4. `std[std = 0] = 1`
5. `X`是稀疏矩阵吗
   - 如果是，要求`zero_center is False `，然后调用`sklearn.utils.sparsefuncs.inplace_column_scale(X, 1 / std)`
   - 如果不是，`zero_center is True`则减`mean`除`std`，否则只除以`std`
6. `X`中大于`max_value`的削到`max_value`
7. 根据`return_mean_value`和`copy`的情况返回值



### _utils

##### _get_mean_var(X, *, axis=0)

计算每列（`axis=0`，对基因求）或每行（`axis=1`，对细胞求）均值和方差（无偏）

### _pca

```python
def pca(
    data: Union[AnnData, np.ndarray, spmatrix],
    n_comps: Optional[int] = None,
    zero_center: Optional[bool] = True,
    svd_solver: str = 'arpack',
    random_state: AnyRandom = 0,
    return_info: bool = False,
    use_highly_variable: Optional[bool] = None,
    dtype: str = 'float32',
    copy: bool = False,
    chunked: bool = False,
    chunk_size: Optional[int] = None,
) -> Union[AnnData, np.ndarray, spmatrix]:
```



## _utils

##### _check_array_function_arguments(**kwargs)

如果`kwargs`里有哪个参数值不是`None`的就报错

##### sanitize_anndata(adata)



### What's New

##### Median absolute deviation

单变量情况下为数据点与中位数的偏差绝对值的中位数
$$
MAD = median(|X_i - median(X)|)
$$

##### Generalized Linear Model

