# nada-lite

### use neo.py to generate
hyperparams:
* `lr1` is learning rate for the `frozen generator`
* `lr2` is learning rate for the `style generator`
* `iteration1` is iteration for the `frozen generator`
* `iteration2` is iteration for the `style generator`
* `dir_lambda` is the weight parameter for the `dir_loss`
* `content_lambda` is the weight parameter for the `content_loss` (not used for now)
* `patch_lambda` is the weight parameter for the `patch_loss`
* `norm_lambda` is the weight parameter for the `norm_loss`
* `gol_lambda` is the weight parameter for the `gol_loss` (not used for now) 

details:
* the `source` and `target` decide the process of generating
* use `content_loss` to train the `frozen generator` first, then continue with the loss function to generate a target pic 

time:
with `iteration1=250` and `iteration2=250`, the cost of generating will be about 180s on RTX2070-maxq

