
<!-- README.md is generated from README.Rmd. Please edit that file -->

# SmoothHOOI

<!-- badges: start -->

<!-- badges: end -->

SmoothHOOI is an R package that implements methods described in “Smooth
Tensor Decomposition with Application to Ambulatory Blood Pressure
Monitoring Data” \[[arxiv](https://arxiv.org/abs/2507.11723)\]. The
application to real and synthetic ABPM data can be found
[here](https://github.com/IrinaStatsLab/SmoothTensorDecompositionABPM).

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
#> [1,] 1.172131 1.166624 1.161571 1.158805 1.156900 1.155208 1.153692 1.152186
#> [2,] 1.234682 1.211747 1.198420 1.189206 1.182093 1.176318 1.171466 1.167292
#> [3,] 1.289497 1.260405 1.238390 1.222483 1.210310 1.200565 1.192720 1.186061
#> [4,] 1.300844 1.259251 1.235752 1.219412 1.207244 1.197742 1.190135 1.186446
#> [5,] 1.273517 1.255178 1.244904 1.238803 1.234318 1.230654 1.227426 1.224509
#> [6,] 1.352194 1.320919 1.301412 1.287224 1.275917 1.266582 1.258701 1.251891
#> [7,] 1.453442 1.397656 1.361187 1.334941 1.314926 1.299083 1.286177 1.275332
#> [8,] 1.500647 1.417691 1.371167 1.340492 1.318204 1.301046 1.287324 1.276008
#>          [,9]    [,10]
#> [1,] 1.150653 1.149140
#> [2,] 1.163660 1.160420
#> [3,] 1.180355 1.175345
#> [4,] 1.180777 1.175867
#> [5,] 1.221764 1.219086
#> [6,] 1.245861 1.240507
#> [7,] 1.266149 1.258212
#> [8,] 1.266500 1.258304
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
#>  [1,] -0.076650096  0.299049554 -0.201243846
#>  [2,] -0.034449943  0.313956989 -0.128558202
#>  [3,]  0.002849221  0.322579412 -0.041857938
#>  [4,]  0.030017644  0.327233531  0.047679747
#>  [5,]  0.042943696  0.321615184  0.132273364
#>  [6,]  0.037863952  0.304783373  0.207738962
#>  [7,]  0.015019838  0.275447492  0.263503164
#>  [8,] -0.022350327  0.234392977  0.295446820
#>  [9,] -0.072918924  0.181367394  0.304222184
#> [10,] -0.132442618  0.118436436  0.292473103
#> [11,] -0.192241118  0.053151088  0.266506091
#> [12,] -0.242459452 -0.005007296  0.234142847
#> [13,] -0.281409555 -0.053541168  0.198182635
#> [14,] -0.304318352 -0.090813772  0.161371961
#> [15,] -0.317274954 -0.113153826  0.117174882
#> [16,] -0.322134647 -0.117725431  0.063340247
#> [17,] -0.320558436 -0.103944161  0.001439115
#> [18,] -0.311771183 -0.070543071 -0.066251142
#> [19,] -0.294631545 -0.019678011 -0.136540452
#> [20,] -0.268631796  0.043723849 -0.199613788
#> [21,] -0.235080563  0.110461774 -0.246343137
#> [22,] -0.197700133  0.172795540 -0.271844798
#> [23,] -0.159784446  0.228687446 -0.272871208
#> [24,] -0.119917417  0.271540854 -0.250302004
tilde$R_tilde
#>            [,1]       [,2]
#> [1,] -0.4216289  0.8839449
#> [2,]  0.1862408 -0.1337787
#> [3,]  0.8874364  0.4480454
tilde$G_tilde[ , ,1]
#>           [,1]      [,2]
#> [1,] 0.5723119 -1.458918
#> [2,] 0.6923224  1.595755
#> [3,] 0.4895840  0.330032
```
