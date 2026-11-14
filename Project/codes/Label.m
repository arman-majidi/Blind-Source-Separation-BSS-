function label = Label(label, thr, rule, sectionNum)
    if (rule == 1)
        label = sectionNum * (label < thr) + 1;
    else
        label = sectionNum * (label > thr) + 1;
    end
end

