clear ; close all;

t = loadMNISTImages("t10k-images.idx3-ubyte")';
y = loadMNISTLabels("t10k-labels.idx1-ubyte");

fprintf("data loaded");

load own_params.mat;

fprintf("params loaded");

sel = randperm(size(t, 1));
sel = sel(1:100);
x_used = t(sel, : );

pred = predict(Theta1, Theta2, t);
pred(pred == 10) = 0;

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

for i = 1:100

	displayData(x_used(i, :));
	p = nn_test(t,sel(i), Theta1, Theta2);
	fprintf("Predicted Value is %f\n", p);
	fprintf("Press Enter key for next test\n");
	pause
	close;

end

