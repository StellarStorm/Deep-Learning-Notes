+++
author = "Skylar"
title = "Region-Overlap Metrics/Losses"
date = "2021-06-16"
tags = ["segmentation"]
categories = [
    "deep learning",
]
+++

Overlap metrics of two regions are excellent ways to quantify how well a model
is able to predict segmentation mappings.

All code examples below are simply psuedo-code that has been simplified for
clarity. For actual loss functions,
[RadIO](https://analysiscenter.github.io/radio/_modules/radio/models/keras/losses.html)
has ready-to-go examples under the Apache 2.0 license.

![intersection](/images/posts/region_loss/set_intersection.png)
![union](/images/posts/region_loss/set_union.png)

*Left: Intersection of two sets. Right: Union of two sets <cite>[1]</cite>*

## Jaccard

The Jaccard index, also called IoU (Intersection over Union), is perhaps one of
the easiest metrics to visualize. It's simply the ratio of the intersection of
two segmented regions, or the number of points that are inside both
segmentations, to the union of two segmented regions, or the number of points
that are inside either (or both) segmentations.

Mathmatically, this looks like

$$ J(A, B) = \frac{|A \cap B|}{|A \cup B|} $$

*Note that $|A|$ indicates the "cardinality" or the number of elements in set
$A$, not the absolute value. This is true for the other sets as well*

This can be rewritten as the ratio between the intersection and the sum of
all points in set $A$ and all points in set $B$ minus the intersection

$$ J(A, B) = \frac{|A \cap B|}{|A| + |B| - |A \cap B|} $$

It may not be immediately clear how this is equivalent - why is intersection
being subtracted? In the denominator, adding all points from set $A$ and set B
means that the points in the intersection $A \cap B$, which are by definition
common to both sets, are added twice. Subtracting one of these intersections
makes the expression in the denominator equivalent to the union of the sets.

That is, the union of sets $A$ and $B$ can be written as

$$\begin{eqnarray}
|A \cup B| &=& {\overbrace{|A - B|}^\text{A complement B}} + |A \cap B| + {\overbrace{|A - B|}^\text{B complement A}} \\\\\\
&=& |A - B| + |B - A| + |A \cap B|
\end{eqnarray}$$

The sum of sets $A$ and $B$ can be written as

$$\begin{eqnarray}
|A| + |B| &=& {\overbrace{|A - B|}^\text{A complement B}} + |A \cap B| + {\overbrace{|A - B|}^\text{B complement A}} + |A \cap B| \\\\\\
&=& |A - B| + |B - A| + \textcolor{red}{2}|A \cap B|
\end{eqnarray}$$

Thus, subtracting the intersection from the summation of sets $A$ and $B$ makes this
equivalent to the union of the sets.

$$A| \cup |B| = |A| + |B| - |A \cap B|$$

The second expression yields itself particularly well to code implementations.
In TensorFlow, this might be written as follows, where y_true is the ground
truth segmentation and y_pred is the model prediction. Both of these must be
boolean tensors.

```python
def jaccard(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = K.sum(y_true * y_pred)

    jac = intersection / (K.sum(y_true + y_pred) - intersection)
    return jac
```

The Jaccard index increases as the two regions have more and more overlap, to
a maximum value of 1 if there is perfect agreement between the two regions, and
decreases to a minimum value of 0 if there is no overlap between the two
regions. Thus, this provides a simple but effective metric for determining how
well the predicted segmentation from a model agrees with the ground truth.

Jaccard loss (also called Jaccard distance) is simply 1 - the jaccard score.
Thus, a value of 0 would indicate perfect agreement, while a value of 1 would
indicate no overlap.

## Dice

The Dice coefficient, also called Sørensen–Dice or F score and abbreviated DSC, is a
second region overlap metric. It is functionally equivalent to the Jaccard
index, in fact, a score in one metric can easily be converted to a score in the
other. DSC is the ratio of twice the intersection of two sets to the number
of elements in each set.

DSC is one of the most common metrics for segmentation models, and, adapted
somewhat, equally as common as a loss function during training.

Mathematically, it can be expressed as

$$S(A, B) = \frac{2 |A \cap B|}{|A| + |B|}$$

*As before, $|A|$ indicates cardinality, not absolute value.*

In Tensorflow, this might be written as follows, where y_true is the ground
truth segmentation and y_pred is the model prediction. Both of these must be
boolean tensors.

```python
def dsc(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = K.sum(y_true * y_pred)

    n_points = K.sum(A) + K.sum(B)

    return 2 * intersection / n_points
```

Like the Jaccard index, possible scores range from 0 to 1, with 1 indicating
perfect agreement and 0 indicated complete disagreement. Thus, turning this
into a loss function would simply be 1 - the DSC score.

### Jaccard vs. Dice

![jaccard_or_dice](/images/posts/region_loss/jac_vs_dsc.png)

*Left: Illustration of Jaccard. Right: Illustration of DSC <cite>[2]</cite>*

Both of these metrics are similar, positively correlated, and can be converted
into each other. However, they emphasize different things.

The Jaccard index tends to penalize single bad examples, giving an estimation
of the "worst case scenario" for a given model. Conversely, DSC gives closer
to an indication of the average performance of a model [3]. The appropriate
choice for training a model (or to use in combination with another type of
loss, such as categorical crossentropy) depends on the project goals and
acceptable outcomes. Intuitively I suspect that a model trained using DSC loss is
more likely to give "homogenous" results, i.e. most predictions are either poor,
ok, or good; wherease one trained using Jaccard loss may give both excellent
and sub-par segmentations.


## References

[1] [Wikipedia](https://en.wikipedia.org/wiki/Jaccard_index)

[2] [Ekin Tiu](https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2)

[3] [willem](https://stats.stackexchange.com/a/276144)
