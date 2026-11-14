function filteredData = Filter(data, fc)
    filteredData = zeros(size(data));
    fs = 2400;
    f = -fs/2: fs/size(data, 2): fs/2 - fs/size(data, 2);
    for i = 1:size(data, 1)
        for j = 1:size(data,3)
            Data = fftshift(fft(data(i, :, j)));
            Data((fc(2) < abs(f)) | (abs(f) < fc(1))) = 0;
            filteredData(i, :, j) = real(ifft(ifftshift(Data)));
        end
    end
end