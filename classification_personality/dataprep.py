# Preprocessing
#==============================================================================

# Detect continuous and categorical variables
likely_cat = {}
for var in df.columns:
    likely_cat[var] = 1. * df[var].nunique() / df[var].count() < 0.05

cats = [var for var in df.columns if likely_cat[var]]
conts = [var for var in df.columns if not likely_cat[var]]

# Remove target from lists
try:
    conts.remove(target)
    cats.remove(target)
except:
    pass

# Convert target to float
df[target] = df[target].apply(pd.to_numeric, errors='coerce').dropna()

print('CATS=====================')
print(cats)
print('CONTS=====================')
print(conts)

# Populate categorical and continuous lists
#==============================================================================

if VARIABLE_FILES == True:
    with open(f'{PARAM_DIR}/cats.txt', 'r') as f:
        cats = f.read().splitlines()

    with open(f'{PARAM_DIR}/conts.txt', 'r') as f:
        conts = f.read().splitlines()

#==============================================================================

# Data cleaning and preprocessing

procs = [Categorify, FillMissing, Normalize]

# Shrink the dataset if necessary
df = df[0:SAMPLE_COUNT]

# Randomly shuffle the data
if SHUFFLE_DATA:
    df = df.sample(frac=1).reset_index(drop=True)

# Workaround for fastai/pytorch bug where bool is treated as object and thus erroring out
for n in df:
    if pd.api.types.is_bool_dtype(df[n]):
        df[n] = df[n].astype('uint8')

# Remove columns listed in cols_to_delete.txt
with open(f'{PARAM_DIR}/cols_to_delete.txt', 'r') as f:
    cols_to_delete = f.read().splitlines()

for col in cols_to_delete:
    try:
        del(df[col])
    except:
        pass

# Try to fill in missing values
try:
    df = df.fillna(0)
except:
    pass

# Convert continuous variables to floats
for var in conts:
    try:
        df[var] = df[var].apply(pd.to_numeric, errors='coerce').dropna()
    except:
        print(f'Could not convert {var} to float.')
        pass

# Experimental logic to add columns one-by-one to find a breakpoint
if ENABLE_BREAKPOINT == True:
    temp_procs = [Categorify, FillMissing]
    print('Looping through continuous variables to find breakpoint')
    cont_list = []
    for cont in conts:
        focus_cont = cont
        cont_list.append(cont)
        try:
            to = TabularPandas(df, procs=procs, cat_names=cats, cont_names=cont_list, y_names=target, y_block=RegressionBlock(), splits=splits)
            del(to)
        except:
            print('Error with ', focus_cont)
            cont_list.remove(focus_cont)
            continue
    for var in cont_list:
        try:
            df[var] = df[var].apply(pd.to_numeric, errors='coerce').dropna()
        except:
            print(f'Could not convert {var} to float.')
            cont_list.remove(var)
            if CONVERT_TO_CAT == True:
                cats.append(var)
            pass
    print(f'Continuous variables that made the cut: {cont_list}')
    print(f'Categorical variables that made the cut: {cats}')
    df = df_shrink(df)

# Creating tabular object + quick preprocessing
to = None
if REGRESSOR == True:
    try:
        to =

