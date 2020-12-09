def regress_out(
    adata: AnnData,
    keys: Union[str, Sequence[str]],
    n_jobs: Optional[int] = None,
    copy: bool = False,
) -> Optional[AnnData]:
    """\
    Regress out (mostly) unwanted sources of variation.
    Uses simple linear regression. This is inspired by Seurat's `regressOut`
    function in R [Satija15]. Note that this function tends to overcorrect
    in certain circumstances as described in :issue:`526`.
    Parameters
    ----------
    adata
        The annotated data matrix.
    keys
        Keys for observation annotation on which to regress on.
    n_jobs
        Number of jobs for parallel computation.
        `None` means using :attr:`scanpy._settings.ScanpyConfig.n_jobs`.
    copy
        Determines whether a copy of `adata` is returned.
    Returns
    -------
    Depending on `copy` returns or updates `adata` with the corrected data matrix.
    """
    start = logg.info(f'regressing out {keys}')
    if issparse(adata.X):
        logg.info(
            '    sparse input is densified and may '
            'lead to high memory use'
        )
    adata = adata.copy() if copy else adata

    sanitize_anndata(adata)

    # TODO: This should throw an implicit modification warning
    if adata.is_view:
        adata._init_as_actual(adata.copy())

    if isinstance(keys, str):
        keys = [keys]

    if issparse(adata.X):
        adata.X = adata.X.toarray()

    n_jobs = sett.n_jobs if n_jobs is None else n_jobs

    # regress on a single categorical variable
    variable_is_categorical = False
    if keys[0] in adata.obs_keys() and is_categorical_dtype(adata.obs[keys[0]]):
        # 如果第一个key是分类变量
        if len(keys) > 1:
            raise ValueError(
                'If providing categorical variable, '
                'only a single one is allowed. For this one '
                'we regress on the mean for each category.'
            )
        logg.debug('... regressing on per-gene means within categories')
        regressors = np.zeros(adata.X.shape, dtype='float32')
        for category in adata.obs[keys[0]].cat.categories:
            mask = (category == adata.obs[keys[0]]).values
            for ix, x in enumerate(adata.X.T):
                regressors[mask, ix] = x[mask].mean()
        variable_is_categorical = True
    # regress on one or several ordinal variables
    else:
        # create data frame with selected keys (if given)
        if keys:
            regressors = adata.obs[keys]
        else:
            regressors = adata.obs.copy()

        # add column of ones at index 0 (first column)
        regressors.insert(0, 'ones', 1.0)
        # 插到第一列，列名是ones，所有值都是1

    len_chunk = np.ceil(min(1000, adata.X.shape[1]) / n_jobs).astype(int)
    n_chunks = np.ceil(adata.X.shape[1] / len_chunk).astype(int)

    tasks = []
    # split the adata.X matrix by columns in chunks of size n_chunk
    # (the last chunk could be of smaller size than the others)
    chunk_list = np.array_split(adata.X, n_chunks, axis=1)
    if variable_is_categorical:
        regressors_chunk = np.array_split(regressors, n_chunks, axis=1)
    for idx, data_chunk in enumerate(chunk_list):
        # each task is a tuple of a data_chunk eg. (adata.X[:,0:100]) and
        # the regressors. This data will be passed to each of the jobs.
        if variable_is_categorical:
            regres = regressors_chunk[idx]
        else:
            regres = regressors
        tasks.append(tuple((data_chunk, regres, variable_is_categorical)))

    if n_jobs > 1 and n_chunks > 1:
        import multiprocessing
        pool = multiprocessing.Pool(n_jobs)
        res = pool.map_async(_regress_out_chunk, tasks).get(9999999)
        pool.close()

    else:
        res = list(map(_regress_out_chunk, tasks))

    # res is a list of vectors (each corresponding to a regressed gene column).
    # The transpose is needed to get the matrix in the shape needed
    adata.X = np.vstack(res).T.astype(adata.X.dtype)
    logg.info('    finished', time=start)
    return adata if copy else None

def _regress_out_chunk(data):
    # data is a tuple containing the selected columns from adata.X
    # and the regressors dataFrame
    data_chunk = data[0]
    regressors = data[1]
    variable_is_categorical = data[2]

    responses_chunk_list = []
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    for col_index in range(data_chunk.shape[1]):

        # if all values are identical, the statsmodel.api.GLM throws an error;
        # but then no regression is necessary anyways...
        if not (data_chunk[:, col_index] != data_chunk[0, col_index]).any():
            responses_chunk_list.append(data_chunk[:, col_index])
            continue

        if variable_is_categorical:
            regres = np.c_[np.ones(regressors.shape[0]), regressors[:, col_index]]
        else:
            regres = regressors
        try:
            result = sm.GLM(data_chunk[:, col_index], regres, family=sm.families.Gaussian()).fit()
            new_column = result.resid_response
        except PerfectSeparationError:  # this emulates R's behavior
            logg.warning('Encountered PerfectSeparationError, setting to 0 as in R.')
            new_column = np.zeros(data_chunk.shape[0])

        responses_chunk_list.append(new_column)

    return np.vstack(responses_chunk_list)

