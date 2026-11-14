function confusionMatrix = Test(data, W_csp1, W_lda1, thr1, rule1, filter1, ...
    W_csp2, W_lda2, thr2, rule2, filter2, W_csp3, W_lda3, thr3, rule3, filter3, trueLabel)
    confusionMatrix = zeros(4);
    data = data - mean(data, [2, 3]);
    filteredData = Filter(data, filter1);
    var_signal = pagemtimes(W_csp1.', filteredData);
    var_num = var(var_signal, 0, 2);
    var_num = reshape(var_num, [], size(filteredData, 3));
    label = W_lda1.' * var_num;
    myLabel = Label(label, thr1, rule1, 3);
    ind = (myLabel == 1);
    % filterdData = data(:, :, ind);
    filteredData = Filter(data(:, :, ind), filter2);
    var_signal = pagemtimes(W_csp2.', filteredData);
    var_num = var(var_signal, 0, 2);
    var_num = reshape(var_num, [], size(filteredData, 3));
    if ~isempty(var_num)
        label = W_lda2.' * var_num;
        myLabel(ind) = Label(label, thr2, rule2, 2);
        ind = (myLabel == 1);
        % filteredData = data(:, :, ind);
    end
    
    filteredData = Filter(data(:, :, ind), filter3);
    var_signal = pagemtimes(W_csp3.', filteredData);
    var_num = var(var_signal, 0, 2);
    var_num = reshape(var_num, [], size(filteredData, 3));
    if ~isempty(var_num)
        label = W_lda3.' * var_num;
        myLabel(ind) = Label(label, thr3, rule3, 1);
    end

    % for i = 1:length(myLabel)
    %     confusionMatrix(trueLabel(i), myLabel(i)) = confusionMatrix(trueLabel(i), myLabel(i)) + 1;
    % end
    inds = sub2ind(size(confusionMatrix), trueLabel, myLabel);
    cnts = accumarray(inds.', 1, size(confusionMatrix(:)));
    confusionMatrix = confusionMatrix + reshape(cnts, size(confusionMatrix));
end