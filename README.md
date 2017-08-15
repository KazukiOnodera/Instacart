# Instacart Market Basket Analysis 2nd place solution

I made two models for predicting reorder & None.
Following are the features I made.

## Features
### User feature
* repeat previous ratio
* order span mean
* time for visiting
* Has ordered orgenic, glutenfree, Asian
* hour delta
* order size
* How many None

### Item feature
* ratio of ordered time
* ordered time cycle
* co-occur(over order)
* stats of pos_cart
* How many user buy it as "one shot"
* stats of number of items co-occured
* stats of order streak
* 1to1 prob
* prob of reorder within N order_number
* distribution delta of dow
* prob of reorder, after first order
* stats of order_number delta

### User x Item feature
* total order
* days since last order
* streak
* stats of pos_cart
* ratio of ordered time
* ordered today
* co-occur
* replacement

### datetime feature
* How many come by dow and hour

More detail, please refer to source.

## F1 maximization
Regarding F1 maximization, I hadn't read that paper until Faron had published the kernel. But I got high score because of my F1 maximization.
Let me explain it.
For maximizing F1, I generate y_true according to predicted prob. And check F1 from higher prob.
For example, lets say we have ordered item and prob, like {A: 0.3, B:0.5, C:0.4}. Then generate y_true in many times. In my case, generated 9999 times.
So now we have many of y_true, like [ [A,B],[B],[B,C],[C],[B],[None].....].
As I mentioned above, next thing we do is to check F1 from [B], [B,C], [B,C,A]. Then we can estimate F1 peak out, and stop calculation, and go next order.
You may know, in this method, we don't need to check all pattern, like [A],[A,B],[A,B,C],[B]...
I guess some might have figured out this method from my comment of "tips to go farther".
However, this method is time consuming as well as depends on seed. So finally I used Faron's kernel. 
Fortunatelly or not, I got almost same result using Faron's kernel.
Please refer to py_model/pyx_get_best_items.pyx

## How to run
Pending

## Requirements
Pending
