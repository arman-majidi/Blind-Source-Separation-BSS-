function filteredData = Filter(data, fc)
    fs = 2400;
    [z, p, k] = butter(7, fc ./ (fs/2), "bandpass");
    sos = zp2sos(z, p, k);
    filteredData = sosfilt(sos, data, 2);
end