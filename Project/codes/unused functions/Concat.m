function result = Concat(data)
    num = length(data);
    [m, n, p] = deal(zeros(1, num));
    for i = 1:num
        [m(i), n(i), p(i)] = size(data{i});        
    end
    result = zeros(m(1), n(1), sum(p));
    p = [1 p];
    for i = 1:num
        result(:, :, p(i):p(i)+p(i+1)-1) = data{i};
    end
end