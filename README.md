# FLAIR Brain Tumor segmentation with DNN: from U-Net architecture to Vision Transformers

This is the repository for the Deep learning course final project@ Units, Trieste, Italy. Here you can find all related files for the solution of the proposed task. 

![image](https://github.com/user-attachments/assets/ed135868-d1b4-4233-a0c1-0fa696333677)



## Project Overview

The project presents diffent approaches to the brain tumor segmentation with images obtained from fluid attenuated inversion recovery. 
Different approaches are proposed, from U-net artificial neural networks to more moder architectures: vision transformers. 

Different loss usages are considered, with a particular attention to the intrinsic relation between result and the objective function. 

> **For a better rendering of the `.ipynb` notebook** (available in the code folder) **please use `Visual Studio Code`.**

## Computational Resources

All the consideration power references should take into consideration that the project was realised entirely on a `MacBook Air M3 (2024) 256 GB`, with limited capabilities. The main library used is `Torch`, with `mps` device.

## Dataset

The used dataset is freely available at [4]. It contains approximatively 4000 images, with a weight of some gigabites. Different dataset could be more appropriated to the cause, but the average weight of 90 GB would have made the training procedure impossible on the hardware available. 

However, you can find there:
* Images $\rightarrow$ Then transformed to `96 x 96 px` with `3` channels (RGB)
* Binary Tumor Mask
  * Empty (`0`) in case of tumor absence
  * with `1` regions in case of tumor

An example could be provided here:

<img width="557" alt="image" src="https://github.com/user-attachments/assets/d4df8947-5e70-48eb-b578-79763fb8dc7c" />

<!-- Notice that, even if the dataset is medical, we have this imbalance between classes:

<img width="406" alt="image" src="https://github.com/user-attachments/assets/ff3920fe-e99a-4646-82b4-379261f63ffa" /> -->


## Proposed models overview

The proposed model for the solution are 

* U-net trained with Dice Loss for approximately 20 epochs

* U-net trained with combo Loss with $0.7 * BCE Loss + 0.3 * Dice Loss$

* U-net trained with combo Loss with $0.5 * BCE Loss + 0.5 * Dice Loss$

And you can find the implementation of the SEgmenter TRansformer (SETR):

* SETR trained with Dice Loss for approximately $350$ epochs $\rightarrow$ $6$ hours trained

Different considerations are made for model definitions and model training. 
Training procedure uses the `adam` optimizer with a small $10^{-4}$ learning rate.

## Results and final considerations

Considering that models are trained on different losses, results are different. Results are accessed with different metrics over the training set (20% dataset).

The vision transformer is the best one from the metrics side.

On the proposed presentation PowerPoint you can also recover different model prediction evolution over the training on a fixed image, in order to see the model convergence towards the real tumoral region.

Lastly, a consideration over model parameter number is made, making it visible as

* All U-net models (since they have same architecture) have approximatively $1.8 M$ parameters

* The SETR model have approximatively $2.5 M$ parameters

* The SETR model with an appropriate large dimension parameter choice would have approximatively $203 M$ parameters, which is clearly impossible to train on the hardware considered above.



## References

>
> `[1]` EU cancer statistics: https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Cancer_statistics
>
> `[2]` Brain tumor segmentation with Deep Neural Networks,Medical Image Analysis, Mohammad Havaei, Axel Davy, and others. https://doi.org/10.1016/j.media.2016.05.004.
>
> `[3]`  Multi-class glioma segmentation on real-world data with missing MRI sequences: comparison of three deep learning algorithms: https://www.nature.com/articles/s41598-023-44794-0
>
> `[4]`Dataset: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
>
> `[5]` Azad, R., Heidary, M., Yilmaz, K., Hüttemann, M., Karimijafarbigloo, S., Wu, Y., Schmeink, A., & Merhof, D. (2023). Loss Functions in the Era of Semantic Segmentation: A Survey and Outlook. arXiv:2312.05391. https://arxiv.org/abs/2312.05391
>
> `[6]` Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. arXiv preprint arXiv:1505.04597. https://arxiv.org/abs/1505.04597
> 
> `[7]` The U-net: A Complete Guide. https://medium.com/@alejandro.itoaramendia/decoding-the-u-net-a-complete-guide-810b1c6d56d8
>
> `[8]` Loss Function library: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Combo-Loss
> 
> `[9]` Image Segmentation Using Vision Transformers (ViT): A Deep Dive with Cityscapes and CamVid Datasets https://medium.com/@ankitrajsh/image-segmentation-using-vision-transformers-vit-a-deep-dive-with-cityscapes-and-camvid-datasets-fc1ccdca295b
>
> `[10]` Segmenter: Transformer for Semantic Segmentation, Robin Strudel and Ricardo Garcia and Ivan Laptev and Cordelia Schmid,
https://arxiv.org/abs/2105.05633



