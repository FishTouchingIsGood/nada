# nada-lite

### use gantest.py to generate
* `lr1` is learning rate for the `frozen generator`
* `lr2` is learning rate for the `style generator`
* `iteration1` is iteration for the `frozen generator`
* `iteration2` is iteration for the `style generator`
* the `source` and `target` decide the process of generating

with `iteration1=1000` and `iteration2=100`, the cost of generating will be about 50s on RTX2070-maxq