
<!-- README.md is generated from README.Rmd. Please edit that file -->

# SmoothHOOI

<!-- badges: start -->

<!-- badges: end -->

SmoothHOOI is an R package that implements methods described in ‘’Smooth
Tensor Decomposition with Application to Ambulatory Blood Pressure
Monitoring Data’’

## Installation

You can install the development version of SmoothHOOI from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("IrinaStatsLab/SmoothHOOI")
```

## Example

This is a simple example which shows you how to use this package.

``` r
library(SmoothHOOI)

set.seed(1234321)

# Generate a random tensor with missing data, in the array format
dims <- c(24, 3, 207)
tnsr <- array(rnorm(prod(dims), mean = 0, sd = 1), dim = dims)

missing_prob <- 0.2
missing_indices <- sample(length(tnsr), size = floor(length(tnsr) * missing_prob))
tnsr[missing_indices] <- NA
```

``` r
# Make the second order difference matrix with circular nature
D2 <- SecDiffMat(24)
```

``` r
# Find optimal hyperparameter with 5-fold cross-validation
kcv_res <- kcv(tnsr, rank_grid=as.matrix(expand.grid(r1<-seq(3,6,by=1), r2<-c(2,3))), lambda_seq=seq(1,10,by=1), k=5, L0=NULL, D=D2, tol=0.01, max_iter=500, init=0)

kcv_res$MSE_mat # matrix for CV error (rows representing ranks, cols representing lambda)
#>          [,1]     [,2]     [,3]     [,4]     [,5]     [,6]     [,7]     [,8]
#> [1,] 1.172131 1.166626 1.161576 1.158826 1.156901 1.155217 1.153676 1.152170
#> [2,] 1.227177 1.205206 1.190116 1.181757 1.175460 1.170243 1.165838 1.162020
#> [3,] 1.278286 1.251227 1.232606 1.218674 1.207836 1.199102 1.191896 1.185804
#> [4,] 1.292700 1.251249 1.226943 1.210377 1.202834 1.193628 1.186046 1.179746
#> [5,] 1.262352 1.252891 1.250023 1.245784 1.242255 1.239026 1.236034 1.233090
#> [6,] 1.345476 1.317767 1.301480 1.288158 1.277483 1.268686 1.261294 1.254862
#> [7,] 1.469506 1.414205 1.376974 1.349758 1.328841 1.312307 1.298713 1.287339
#> [8,] 1.465609 1.392685 1.351531 1.323935 1.303719 1.288109 1.275645 1.265289
#>          [,9]    [,10]
#> [1,] 1.150657 1.149116
#> [2,] 1.158628 1.155584
#> [3,] 1.180579 1.176029
#> [4,] 1.174367 1.169655
#> [5,] 1.230345 1.227648
#> [6,] 1.249189 1.244159
#> [7,] 1.277729 1.269320
#> [8,] 1.256500 1.248908
kcv_res$opt_para # optimal hyperparameters
#>      [,1] [,2] [,3]
#> [1,]    3    2   10
```

``` r
# Run SmoothHOOI algorithm with the optimal hyperparameters
res <- mglram(tnsr, ranks = c(3, 2), init=0, D = D2,
       lambda = 10, max_iter = 500, tol = 1e-5, L0 = NULL)

res$conv # check convergence
#> [1] TRUE
```

``` r
# Rotation for Identifiability
tilde <- MakeIdent(L=res$L, G=res$G, R=res$R)

tilde$L_tilde
#>               [,1]         [,2]         [,3]
#>  [1,] -0.076664031  0.299039188 -0.201257836
#>  [2,] -0.034462738  0.313950831 -0.128575363
#>  [3,]  0.002837489  0.322577973 -0.041876176
#>  [4,]  0.030006785  0.327236749  0.047662495
#>  [5,]  0.042933955  0.321622435  0.132258788
#>  [6,]  0.037855655  0.304793481  0.207728255
#>  [7,]  0.015013668  0.275458956  0.263496525
#>  [8,] -0.022353592  0.234403947  0.295443612
#>  [9,] -0.072918581  0.181376161  0.304221700
#> [10,] -0.132438500  0.118441802  0.292474823
#> [11,] -0.192233673  0.053152657  0.266509439
#> [12,] -0.242449631 -0.005008934  0.234147329
#> [13,] -0.281398641 -0.053545019  0.198188052
#> [14,] -0.304307991 -0.090818852  0.161378679
#> [15,] -0.317266471 -0.113160101  0.117183434
#> [16,] -0.322129268 -0.117733398  0.063350895
#> [17,] -0.320556710 -0.103954596  0.001451675
#> [18,] -0.311773461 -0.070556550 -0.066237379
#> [19,] -0.294637784 -0.019694494 -0.136527093
#> [20,] -0.268641612  0.043705177 -0.199602787
#> [21,] -0.235093174  0.110442486 -0.246335905
#> [22,] -0.197714498  0.172777116 -0.271842409
#> [23,] -0.159799543  0.228670861 -0.272874407
#> [24,] -0.119932297  0.271526977 -0.250310897
tilde$R_tilde
#>            [,1]       [,2]
#> [1,] -0.4215993  0.8839439
#> [2,]  0.1862318 -0.1338599
#> [3,]  0.8874524  0.4480230
tilde$G_tilde[ , ,1]
#>           [,1]      [,2]
#> [1,] 0.5721879 -1.459163
#> [2,] 0.6924006  1.595611
#> [3,] 0.4895365  0.329901
```
