function p = nn_test (X,i, Theta1, Theta2)
	a1 = [1 X(i,:)];
	z2 = (Theta1*a1')';
	a2 = sigmoid(z2);
	a2 = [ones(size(a2,1),1) a2];
	z3 = (Theta2*a2')';
	a3 = sigmoid(z3);
	
	[_ p] = max(a3);
end
