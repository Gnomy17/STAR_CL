# :star2: STAR: Stability Inducing Weight Perturbation for Continual Learning

Official Repository of [STAR: Stability Inducing Weight Perturbation for Continual Learning (ICLR 2025)](https://openreview.net/forum?id=6N5OM5Duuj)

### :dizzy: **STAR is now an official part of the [mammoth](https://github.com/aimagelab/mammoth/tree/master/datasets) library.** :dizzy:
You can either use this repository, or the [mammoth](https://github.com/aimagelab/mammoth/tree/master/datasets) library to use STAR. The instructions below for _adding STAR to your own code_ and the arguments for running the experiments should work in both repositories!

## Presentation

### Poster

![30891](https://github.com/user-attachments/assets/f2f8063d-5a54-41e5-8465-19eb795ded77)

### Video

https://github.com/user-attachments/assets/68fa1afb-ffb1-414c-85d3-9718796faffa



This library was built upon [mammoth](https://github.com/aimagelab/mammoth/tree/master/datasets).

The main part of our codes is in `model/utils/star_perturber.py`. Instructions on adding it to any model in mammoth are at the end of this document.

## Running Experiments

You can use the following commands to run each experiment with the best arguments for CIFAR10 with 500 buffer size, arguments for other datasets can be found in the supplementary material pdf.

```
##### DER++ + STAR
python utils/main.py --alpha=0.2 --beta=0.1 --buffer_size=100 --dataset=seq-cifar10 --lr=0.1 --model=derpp_star --p-gamma=0.1 --p-lam=0.005

##### ER-ACE + STAR
python utils/main.py --buffer_size=100 --dataset=seq-cifar10 --lr=0.03 --model=er_ace_star --p-gamma=0.01 --p-lam=0.1

##### ER + STAR
python utils/main.py --buffer_size=100 --dataset=seq-cifar10 --lr=0.1 --model=er_star --p-gamma=0.01 --p-lam=0.1

##### X-DER-RPC + STAR
python utils/main.py --alpha=0.1 --beta=0.5 --buffer_size=100 --dataset=seq-cifar10 --lr=0.03 --model=xder_rpc_star --p-gamma=0.001 --p-lam=0.01
```

## Adding STAR to your own code

Adding STAR to your own method is very easy and only takes a few lines of code. Just follow these steps (assuming you're using mammoth as the base repository)

1. Add `star_perturber.py` to `models/utils/` folder

2. Import STAR and its arguments 
```
from models.utils.star_perturber import Perturber, add_perturb_args
```

3. Add arguments for STAR in `get_parser` function (for example in `er.py`)
```
add_perturb_args(parser)
```

4. Initialize STAR in your model's `__init__` function (for example in `er.py`)
```
self.pert = Perturber(self)
```

5. Call STAR on the buffer data AFTER opt.zero_grad and BEFORE other gradient computations to avoid problems with pytorch autograd
```
self.opt.zero_grad()
if not self.buffer.is_empty():
    # STAR here
    buf_inputs, buf_labels = self.buffer.get_data( 
        self.args.minibatch_size, transform=self.transform)
    self.pert(buf_inputs, buf_labels)    
# the rest of your method
``` 

Keep in mind `self.pert()` adds the STAR gradient by calling `loss.backwards()` on the KL loss function based on the perturbed parameters. This way, the gradients for STAR are added when `opt.step()` is called. Check `utils/star_perturber.py` for more details.

# Cite our work
```
@inproceedings{
eskandar2025star,
title={{STAR}: Stability-Inducing Weight Perturbation for Continual Learning},
author={Masih Eskandar and Tooba Imtiaz and Davin Hill and Zifeng Wang and Jennifer Dy},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=6N5OM5Duuj}
}
```
