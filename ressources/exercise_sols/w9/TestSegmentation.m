% Testing of abdominal segmentation algorithms
clc; clear; close all; 

% Training image
ct_image = '1-105.dcm';
labels = 'label0002.nii';
imageName = 'training.png';

% Validation image
% ct_image = '1-083.dcm';
% labels = 'label0003.nii';
% imageName = 'validation.png';

info = dicominfo(ct_image);
ct = dicomread(ct_image);

% ground truth labels
anno = niftiread(labels);
slice_number = info.InstanceNumber;
Ianno = (uint8(anno(:, :, slice_number)))';

[ISpleen, ILiver, IKidney] = AbdominalSegmentation(ct);

Isegmentation_combined = ILiver + 2 * ISpleen + 3 * IKidney;
RGB_segm = label2rgb(Isegmentation_combined,'spring','w','shuffle'); 

liver_label = 6;
spleen_label = 1;
kidney_label = 3;
ILiver_GT = (Ianno == liver_label);
ISpleen_GT = (Ianno == spleen_label);
IKidney_GT = (Ianno == kidney_label);

Ianno_GT_combined = ILiver_GT + 2 * ISpleen_GT + 3 * IKidney_GT;
RGB_anno = label2rgb(Ianno_GT_combined,'spring','w','shuffle'); 

liver_dice = dice(ILiver_GT, ILiver);
spleen_dice = dice(ISpleen_GT, ISpleen);
kidney_dice = dice(IKidney_GT, IKidney);

str=sprintf('Image %s DICE scores: Liver %g, spleen %g, left kidney %g', ct_image, liver_dice, spleen_dice, kidney_dice);

figure;
subplot(2,2,1); imshow(ct, [-100, 200]);  title('Input image')
subplot(2,2,2);imagesc(RGB_anno); axis image; title('Ground truth segmentations')
subplot(2,2,3);imagesc(RGB_segm); axis image; title('Found segmentations')
subplot(2,2,4);imshowpair(Isegmentation_combined > 0, Ianno_GT_combined > 0); axis image; title('Comparison')
annotation('textbox', [0.05 0.26 0.3 0.3],'String', str,'FitBoxToText', 'on');
saveas(gcf,imageName);