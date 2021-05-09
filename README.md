# Spiking neural WCST model

This repository contains the source code for the spiking neural model of the WCST implemented in Nengo, and published in Kajić, I. and Stewart, T.C (2021): Biologically Constrained Large-Scale Model of the Wisconsin Card Sorting Test (to appear in Proceedings of the 43th Annual Meeting of the Cognitive Science Society).

This code has been tested with Python 3.8.5.

## Getting started

Start by cloning this repository and installing the Python packages in `requirements.txt`:

```
pip install -r requirements.txt
```

From the cloned repository, import the model and run it: 
```python
from model import WCSTModel
import pandas as pd

result = WCSTModel().run(T=5, d=64, x_seq_correct=3, x_deck_size=16)
```
This will create a model with 64-dimensional semantic pointers and run it for 5 seconds. As well, it automatically creates an experimental module under the hood that controls the experiment logistics: `x_seq_correct` says that the rule will change after a sequence of 3 correct responses, and `x_deck_size` determines how many cards there are in the deck. This module can be thought of as the experiment administrator recording the responses and providing feedback.

After at most 10 minutes of building and running the model, we get the results and pretty-print them:

```python
pd.DataFrame(result)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>r_tstart</th>
      <th>r_tend</th>
      <th>trial</th>
      <th>match</th>
      <th>stimulus</th>
      <th>target</th>
      <th>similarity</th>
      <th>choice</th>
      <th>rule</th>
      <th>rule_seq_id</th>
      <th>correct</th>
      <th>n_categories</th>
      <th>error</th>
      <th>p_error</th>
      <th>p_response</th>
      <th>fail_shift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.20</td>
      <td>0.50</td>
      <td>1</td>
      <td>SN</td>
      <td>Y-SQ-ONE</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>N</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>1.10</td>
      <td>2</td>
      <td>SN</td>
      <td>B-SQ-ONE</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>N</td>
      <td>12</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.40</td>
      <td>1.70</td>
      <td>3</td>
      <td>SN</td>
      <td>R-CR-TWO</td>
      <td>2</td>
      <td>1.0</td>
      <td>2</td>
      <td>N</td>
      <td>12</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.00</td>
      <td>2.30</td>
      <td>4</td>
      <td>S</td>
      <td>R-CR-THREE</td>
      <td>2</td>
      <td>1.0</td>
      <td>2</td>
      <td>S</td>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.60</td>
      <td>2.90</td>
      <td>5</td>
      <td>S</td>
      <td>B-ST-ONE</td>
      <td>4</td>
      <td>1.0</td>
      <td>4</td>
      <td>S</td>
      <td>12</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.21</td>
      <td>3.50</td>
      <td>6</td>
      <td>CS</td>
      <td>Y-ST-ONE</td>
      <td>4</td>
      <td>1.0</td>
      <td>4</td>
      <td>S</td>
      <td>12</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.81</td>
      <td>4.11</td>
      <td>7</td>
      <td>CS</td>
      <td>B-CR-ONE</td>
      <td>2</td>
      <td>1.0</td>
      <td>2</td>
      <td>C</td>
      <td>12</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4.41</td>
      <td>4.71</td>
      <td>8</td>
      <td>SN</td>
      <td>R-TR-THREE</td>
      <td>1</td>
      <td>1.0</td>
      <td>3</td>
      <td>C</td>
      <td>12</td>
      <td>X</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Getting serious


Ideally, the model should be run with the `nengo_ocl` backend to take advantage of GPU optimizations that speed up simulation run times. Results presented in the paper were generated with simulations ran that way. The script `run-all.py` shows an example of configuration for running large-scale simulations.  The [nengo_ocl](https://github.com/nengo-labs/nengo-ocl) repository has installation instructions for those wishing to experiment with this setup.

As well, to run the `run-all` script one needs to manually check out the [ocl-use-context](https://github.com/ctn-waterloo/ctn_benchmarks/tree/ocl-use-context) branch, which allows `nengo_ocl` to be provided as the backend.
