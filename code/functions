def get_cox_pred(preds,labels):
    """ function to fit two models,

        'slope model':
        logit(Y) = a' + b * L
        to retreive the calibration slope 'b'

        'intercept model':
        logit(Y) = a + offset(L) 
        (fixing b at 1 by entering L as offset term)
        to retreive the calibration intercept 'a'

    Args:
        preds (list / np.array / pd.Series): predictions
        labels (list / np.array / pd.Series): corresponding ground truth labels

    Returns:
        _type_: _description_
    """

    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    preds[preds==0] = 0.0001
    preds[preds==1] = 0.9999
 
    df = pd.DataFrame()
    df['y'] = labels
    df['logitp'] = logit(preds)
    
    # fit slope model 
    formula = 'y ~ logitp'  
    model = smf.glm(formula = formula, data=df, family=sm.families.Binomial())
    result = model.fit()
    b = result.params[1]
    # 95% CI
    b_low = result.conf_int(alpha=0.05)[0][1]
    b_high = result.conf_int(alpha=0.05)[1][1] 
    
    # save intercept following from the slope model 
    a_prime = result.params[0] 
    # 95% CI
    a_prime_low = result.conf_int(alpha=0.05)[0][1]
    a_prime_high = result.conf_int(alpha=0.05)[1][1] 
    
    # fit intercept model (slope set to unity by entering offset term)
    formula = 'y ~ 1'
    model = smf.glm(formula = formula, data=df, family=sm.families.Binomial(),offset=df['logitp'])
    result = model.fit()

    # save intercept following from the intercept model 
    a = result.params[0] 
    # 95% CI
    a_low = result.conf_int(alpha=0.05)[0][1]
    a_high = result.conf_int(alpha=0.05)[1][1] 
        
    return a, a_low, a_high, b, b_low, b_high, a_prime, a_prime_low, a_prime_high

def logit(p):
    return np.log(p) - np.log(1 - p)

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))


def model_recalibration(model,a,b,a_prime,method):
    """ function to update logistic regression model for recalibration.

    Args:
        model (obj): fitted logistic regression model
        a (float): intercept of intercept model
        b (float): slope of slope model
        a_prime (float): intercept of slope model
        method (str): option to update the model by adjusting just the intercept or the intercept and slope
    """
    
    print('Original calibration intercept and slope: ')
    print('intercept: ',model.intercept_)
    print('slope: ',model.coef_)
    
    if method == 'intercept_only':
        # update intercept using calibration intercept
        model.intercept_ = np.array([model.intercept_ + a])
        
    elif method == 'intercept_and_slope':
        # update intercept using calibration intercept
        model.intercept_ = np.array([model.intercept_ + a_prime])
        # update coefficients using calibration slope
        model.coef_ = clf.coef_*b

    print('Calibration intercept and slope after update: ')
    print('intercept: ',model.intercept_)
    print('slope: ',model.coef_)
    
    return model



def calibration_plot(preds,labels):
    
    from sklearn.calibration import calibration_curve
    from sklearn.utils import resample

    colors= ['tab:blue', 'tab:purple', 'tab:red', 'tab:green']
    strategy='uniform'

    plt.figure(figsize=(29, 25))

    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax3 = plt.subplot2grid((3, 1), (2, 0))
    ax3.set_yscale("log")

    # perfect
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated",linewidth=3)

    # AUROC calculation
    print('discrimination:')
    auc = calc_auc(preds,labels,500) 
    print(auc)
    
    # get COX intercept/slope
    a, a_low, a_high, b, b_low, b_high, a_prime, a_prime_low, a_prime_high = get_cox_pred(preds,labels)
    
    # make 95% CIs
    fracs = []
    mean = np.linspace(0, 1, 100)
    B = 100 #number of bootstraps for 95% CI around calibration curve
    n_bins = 50
    
    for b in range(B):
        pred_boot, labels_boot = resample(preds, labels, stratify=labels, random_state=b)
        fraction_of_positives, mean_predicted_value = calibration_curve(labels_boot, pred_boot, n_bins=n_bins,strategy=strategy)
        
        y = lowess_bell_shape_kern(mean_predicted_value, fraction_of_positives)
        interp_frac = np.interp(mean, mean_predicted_value, y)
        fracs.append(interp_frac)
        
    fracs_lower = np.percentile(fracs,2.5,axis=0)
    fracs_upper = np.percentile(fracs, 97.5,axis=0)
    
    
    # make main calibration line
    fraction_of_positives, mean_predicted_value = \
            calibration_curve(labels, preds, n_bins=n_bins,strategy=strategy)
    
    y = lowess_bell_shape_kern(mean_predicted_value, fraction_of_positives)
    
    
    
    ax1.plot(mean_predicted_value, y, colors[0],linestyle='-',linewidth = 1.5,
                label= "{a}\nAUROC: {k} ({l};{m})\nIntercept: {b} ({w};{x})\nSlope: {c} ({y};{z})".format(a='model',
                                                        w= str(np.round(a_low,2)),
                                                                                
                                                        b=str(np.round(a,2)),x=str(np.round(a_high,2)),
                                                        c=str(np.round(b,2)), y=str(np.round(b_low,2)) , 
                                                        z=str(np.round(b_high,2)),
                                                        k=str(np.round(auc[0],2)),
                                                        l=str(np.round(auc[1][0],2)),
                                                        m=str(np.round(auc[1][1],2)))
                )
    
    
    
    ax1.fill_between(mean, fracs_lower, fracs_upper, color=colors[0], alpha=.2,
                        where=(mean>=np.min(mean_predicted_value))&(mean<=np.max(mean_predicted_value))
                    # label=r'$\pm$ 1 std. dev.'
                    )
    
    ax3.hist(preds, range=(0, 1), bins=n_bins,linestyle='-',linewidth = 1.5,
                label = models[0],
                  histtype="step", lw=2,color=colors[0])


    ax1.set_ylabel("Observed proportion",fontsize=30)
    ax1.set_ylim([-0.01, 1.05])
    ax1.set_xlim([-0.01, 1.05])
    ax1.set_xlabel("Predicted probability",fontsize=30)


    ax1.legend(loc="lower right",prop={'size':25})


    ax1.tick_params(labelsize=20)
    ax3.tick_params(labelsize=20)

    ax3.set_xlabel("Predicted probability",fontsize=33)
    ax1.set_ylabel("Observed proportion",fontsize=33)
    ax3.set_ylabel("Count",fontsize=33)
    ax3.legend(loc="upper center",  prop={'size':25}, ncol=3)

    ax1.tick_params(axis="x", labelsize=28)
    ax1.tick_params(axis="y", labelsize=28)

    ax3.tick_params(axis="x", labelsize=28)
    ax3.tick_params(axis="y", labelsize=28)




def calc_auc(preds,labels,nboot):
    from sklearn.utils import resample
    from scipy.stats import rankdata
    from sklearn import metrics
    from sklearn.metrics import roc_auc_score
    from sklearn import metrics
    
    df = pd.DataFrame()
    df['label'] = labels
    df['pred'] = preds
    
    AUC = roc_auc_score(labels,preds)
    
    
    AUC_boot = []
    for i in range(nboot):
        
        pred_boot, labels_boot = resample(preds, labels, stratify=labels, random_state=i)
        AUC_boot.append(roc_auc_score(labels_boot, pred_boot))
        
    AUC_95CI = [np.percentile(AUC_boot, 2.5, axis=0),np.percentile(AUC_boot, 97.5, axis=0)]
    
    return AUC, AUC_95CI

# functions needed to plot dflexible calibration curve
#Defining the bell shaped kernel function - used for plotting later on
def kernel_function(xi,x0,tau= .005): 
    return np.exp( - (xi - x0)**2/(2*tau)   )

def lowess_bell_shape_kern(x, y, tau = .005):
    """lowess_bell_shape_kern(x, y, tau = .005) -> yest
    Locally weighted regression: fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The kernel function is the bell shaped function with parameter tau. Larger tau will result in a
    smoother curve. 
    """
    n = len(x)
    yest = np.zeros(n)

    #Initializing all weights from the bell shape kernel function    
    w = np.array([np.exp(- (x - x[i])**2/(2*tau)) for i in range(n)])     
    
    #Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        theta = linalg.solve(A, b)
        yest[i] = theta[0] + theta[1] * x[i] 

    return yest