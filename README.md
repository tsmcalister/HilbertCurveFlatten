# Image classification with 1D Convolutions after Hilbert Curve based Flattening

This contains an experiment to see what influence a particular flatten layer has on image classification tasks.


## Hilbert Curve Flatten (HCFlatten)

Given an image of size `(n, n, 3)` where `n = 2^k` for some `k in N, k > 1` the HCFlatten layer defines a mapping
`H_n: N**2 -> N` such that:

```

                               A---B   C---D
                                   |   |    
                               E---F   G---H
                HCFlatten      |           |   --->   A-B-F-E-I-M-N-J-K-O-P-L-H-G-C-D
                               I   J---K   L
                               |   |   |   | 
                               M---N   O---P
    
    
    
                               A---B---C---D
                               ┌-----------┘
                               E---F---G---H
                 Flatten       ┌-----------┘   --->   A-B-C-D-E-F-G-H-I-J-K-L-M-N-O-P
                               I---J---K---L
                               ┌-----------┘
                               M---N---O---P

```

## Experimental setup

## Results
