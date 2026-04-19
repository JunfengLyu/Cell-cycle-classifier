function cell_distinguisher_dataset
clc; close all;

configs = {
    './Dataset_raw/20times',  './Dataset_20times',  160, 1.00;
    './Dataset_raw/100times', './Dataset_100times', 800, 1.00
};

for c = 1:size(configs,1)
    inputDir     = configs{c,1};
    outputDir    = configs{c,2};
    boxSize      = configs{c,3};
    displayGamma = configs{c,4};
    historyFile  = fullfile(inputDir, 'history.tsv');

    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    pairs = find_image_pairs(inputDir);
    if isempty(pairs)
        fprintf('No valid image pairs found in %s\n', inputDir);
        continue;
    end

    doneIDs = load_history_ids(historyFile);
    redoProcessed = true;
    if ~isempty(doneIDs)
        redoProcessed = confirm_yes_no(sprintf('Reprocess handled groups in %s ? [Y/N]: ', inputDir));
    end

    if ~redoProcessed
        keep = true(1, numel(pairs));
        for i = 1:numel(pairs)
            if any(strcmp(doneIDs, pairs(i).id))
                keep(i) = false;
            end
        end
        pairs = pairs(keep);
    end

    if isempty(pairs)
        fprintf('No image groups to process in %s\n', inputDir);
        continue;
    end

    saveCounter = next_cell_index(outputDir);

    for k = 1:numel(pairs)
        dna = read_gray(pairs(k).dnaPath);
        tub = read_gray(pairs(k).tubPath);

        if ~isequal(size(dna), size(tub))
            fprintf('Skip %s: size mismatch\n', pairs(k).rawID);
            continue;
        end

        tub8 = normalize_image(tub, displayGamma, 1, 99.5);

        if boxSize == 160
            dnaMain = normalize_dna_20x(dna);
        else
            dnaMain = normalize_image(dna, 1.05, 1, 98.8);
        end

        mergedMain = zeros([size(dnaMain), 3], 'uint8');
        mergedMain(:,:,3) = dnaMain; % blue
        mergedMain(:,:,2) = tub8;    % green

        baseImg = mergedMain;

        fig = figure('Color', 'w', 'Name', pairs(k).rawID);
        ax = axes('Parent', fig);
        imshow(baseImg, 'Parent', ax);
        title(ax, {['Image group: ' pairs(k).rawID], ...
            'Move mouse: preview box', ...
            'Left click: select one cell', ...
            'Press Enter: finish this image'});
        hold(ax, 'on');

        xlim(ax, [0.5, size(dna,2)+0.5]);
        ylim(ax, [0.5, size(dna,1)+0.5]);

        half = boxSize / 2;
        hHover = rectangle(ax, ...
            'Position', [1-half, 1-half, boxSize, boxSize], ...
            'EdgeColor', 'c', 'LineWidth', 1.2, 'LineStyle', '--', ...
            'Visible', 'off');
        hHoverCenter = plot(ax, 1, 1, 'c+', ...
            'LineWidth', 1.2, 'MarkerSize', 8, 'Visible', 'off');

        set(fig, 'WindowButtonMotionFcn', ...
            @(src,evt) hover_box(src, ax, hHover, hHoverCenter, boxSize, size(dna)));

        while true
            [x, y, isEnter] = get_one_click_or_enter(fig, ax);
            if isEnter || isempty(x)
                break;
            end

            cx = x;
            cy = y;

            if ~box_inside(size(dna), cx, cy, boxSize)
                msgbox('Crop box exceeds boundary. Skipped.', 'Warning', 'warn');
                figure(fig);
                continue;
            end

            hTemp = rectangle(ax, ...
                'Position', [cx-half, cy-half, boxSize, boxSize], ...
                'EdgeColor', 'c', 'LineWidth', 1.5, 'LineStyle', '--');
            hTempCenter = plot(ax, cx, cy, 'c+', 'LineWidth', 1.5, 'MarkerSize', 8);
            drawnow;

            tubCrop = crop_cell(tub, cx, cy, boxSize);
            dnaCrop = crop_cell(dna, cx, cy, boxSize);

            [keepCell, phase] = review_cell(tubCrop, dnaCrop, displayGamma, boxSize);
            figure(fig);

            if ~keepCell
                if isgraphics(hTemp), delete(hTemp); end
                if isgraphics(hTempCenter), delete(hTempCenter); end
                continue;
            end

            prefix = sprintf('%04d_%s', saveCounter, phase);
            imwrite(normalize_image(tubCrop, 1.0, 1, 99.5), fullfile(outputDir, [prefix '_Tubulin.png']));
            if boxSize == 160
                imwrite(normalize_dna_20x(dnaCrop), fullfile(outputDir, [prefix '_DNA.png']));
            else
                imwrite(normalize_image(dnaCrop, 1.05, 1, 98.8), fullfile(outputDir, [prefix '_DNA.png']));
            end

            append_history_record(historyFile, pairs(k).rawID, phase, round(cx), round(cy), ...
                boxSize, prefix, inputDir, outputDir);

            if isgraphics(hTemp), delete(hTemp); end
            if isgraphics(hTempCenter), delete(hTempCenter); end

            rectangle(ax, ...
                'Position', [cx-half, cy-half, boxSize, boxSize], ...
                'EdgeColor', 'y', 'LineWidth', 1.5);
            plot(ax, cx, cy, 'y+', 'LineWidth', 1.5, 'MarkerSize', 8);
            text(ax, cx, cy, num2str(saveCounter), ...
                'Color', 'w', 'FontWeight', 'bold', ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle');

            saveCounter = saveCounter + 1;
        end

        mark_group_done(historyFile, pairs(k).id);

        if isvalid(fig)
            close(fig);
        end
    end
end
end


function hover_box(fig, ax, hHover, hHoverCenter, boxSize, imgSize)
if ~isgraphics(fig) || ~isgraphics(ax)
    return;
end

cp = get(ax, 'CurrentPoint');
x = cp(1,1);
y = cp(1,2);

w = imgSize(2);
h = imgSize(1);

inside = x >= 1 && x <= w && y >= 1 && y <= h;
if ~inside
    set(hHover, 'Visible', 'off');
    set(hHoverCenter, 'Visible', 'off');
    return;
end

half = boxSize / 2;
set(hHover, 'Position', [x-half, y-half, boxSize, boxSize], 'Visible', 'on');
set(hHoverCenter, 'XData', x, 'YData', y, 'Visible', 'on');
end


function [x, y, isEnter] = get_one_click_or_enter(fig, ax)
x = [];
y = [];
isEnter = false;

while true
    waitforbuttonpress;
    if ~isgraphics(fig) || ~isgraphics(ax)
        isEnter = true;
        return;
    end

    key = get(fig, 'CurrentKey');
    sel = get(fig, 'SelectionType');

    if strcmp(key, 'return') || strcmp(key, 'enter')
        isEnter = true;
        return;
    end

    if strcmp(sel, 'normal')
        cp = get(ax, 'CurrentPoint');
        x = cp(1,1);
        y = cp(1,2);
        return;
    end
end
end


function pairs = find_image_pairs(inputDir)
exts = {'*.png','*.jpg','*.jpeg','*.tif','*.tiff','*.bmp'};
files = [];
for i = 1:numel(exts)
    files = [files; dir(fullfile(inputDir, exts{i}))];
end

dnaMap = struct();
tubMap = struct();
rawMap = struct();

for i = 1:numel(files)
    [~, stem, ext] = fileparts(files(i).name);
    tokDNA = regexp(stem, '^(.*)_DNA$', 'tokens', 'once', 'ignorecase');
    tokTUB = regexp(stem, '^(.*)_Tubulin$', 'tokens', 'once', 'ignorecase');

    if ~isempty(tokDNA)
        rawID = tokDNA{1};
        id = matlab.lang.makeValidName(rawID);
        dnaMap.(id) = fullfile(files(i).folder, [stem ext]);
        rawMap.(id) = rawID;
    elseif ~isempty(tokTUB)
        rawID = tokTUB{1};
        id = matlab.lang.makeValidName(rawID);
        tubMap.(id) = fullfile(files(i).folder, [stem ext]);
        rawMap.(id) = rawID;
    end
end

f1 = fieldnames(dnaMap);
f2 = fieldnames(tubMap);
fc = intersect(f1, f2);

pairs = struct('id', {}, 'rawID', {}, 'dnaPath', {}, 'tubPath', {});
for i = 1:numel(fc)
    pairs(end+1).id = fc{i};
    pairs(end).rawID = rawMap.(fc{i});
    pairs(end).dnaPath = dnaMap.(fc{i});
    pairs(end).tubPath = tubMap.(fc{i});
end

[~, idx] = sort(lower({pairs.rawID}));
pairs = pairs(idx);
end


function img = read_gray(path)
img = imread(path);
if ndims(img) == 3
    img = rgb2gray(img);
end
img = double(img);
end


function img8 = normalize_image(img, gammaVal, lowPct, highPct)
if nargin < 2
    gammaVal = 1.0;
end
if nargin < 3
    lowPct = 1;
end
if nargin < 4
    highPct = 99.5;
end

p1 = prctile(img(:), lowPct);
p99 = prctile(img(:), highPct);
if p99 <= p1
    p99 = p1 + 1;
end

img = (double(img) - p1) / (p99 - p1);
img = min(max(img, 0), 1);
img = img .^ gammaVal;
img8 = im2uint8(img);
end


function img8 = normalize_dna_20x(img)
blackPct = 30;
whitePct = 99;
gammaVal = 0.9;

pBlack = prctile(img(:), blackPct);
pWhite = prctile(img(:), whitePct);
if pWhite <= pBlack
    pWhite = pBlack + 1;
end

img = (double(img) - pBlack) / (pWhite - pBlack);
img = min(max(img, 0), 1);
img = img .^ gammaVal;
img8 = im2uint8(img);
end


function tf = box_inside(imgSize, cx, cy, boxSize)
h = imgSize(1);
w = imgSize(2);
half = floor(boxSize/2);
x1 = round(cx - half);
y1 = round(cy - half);
x2 = x1 + boxSize - 1;
y2 = y1 + boxSize - 1;
tf = x1 >= 1 && y1 >= 1 && x2 <= w && y2 <= h;
end


function crop = crop_cell(img, cx, cy, boxSize)
half = floor(boxSize/2);
x1 = round(cx - half);
y1 = round(cy - half);
x2 = x1 + boxSize - 1;
y2 = y1 + boxSize - 1;
crop = img(y1:y2, x1:x2);
end


function [keep, phase] = review_cell(tubCrop, dnaCrop, gammaVal, boxSize)
tub8 = normalize_image(tubCrop, gammaVal, 1, 99.5);
if boxSize == 160
    dna8 = normalize_dna_20x(dnaCrop);
else
    dna8 = normalize_image(dnaCrop, 1.05, 1, 98.8);
end

merged = zeros([size(dna8),3], 'uint8');
merged(:,:,3) = dna8; % blue
merged(:,:,2) = tub8; % green

f = figure('Color','w','Name','Candidate preview');
subplot(1,3,1); imshow(merged); title('Merged');
subplot(1,3,2); imshow(tub8,[]); title('Tubulin');
subplot(1,3,3); imshow(dna8,[]); title('DNA');

while true
    s = upper(strtrim(input('Keep this cell? [Y/N]: ','s')));
    if any(strcmp(s, {'Y','N'}))
        keep = strcmp(s, 'Y');
        break;
    end
    disp('Please input Y or N.');
end

phase = '';
if keep
    valid = {'I','P','M','A','T'};
    while true
        phase = upper(strtrim(input('Cell-cycle phase [I/P/M/A/T]: ','s')));
        if any(strcmp(phase, valid))
            break;
        end
        disp('Please input I, P, M, A, or T.');
    end
end

if isvalid(f)
    close(f);
end
end


function doneIDs = load_history_ids(historyFile)
doneIDs = {};
if ~exist(historyFile, 'file')
    return;
end

T = readtable(historyFile, 'FileType', 'text', 'Delimiter', '\t');
if isempty(T)
    return;
end

if any(strcmp(T.Properties.VariableNames, 'type')) && any(strcmp(T.Properties.VariableNames, 'raw_id'))
    idx = strcmp(string(T.type), "done_group");
    doneIDs = unique(cellstr(string(T.raw_id(idx))));
    doneIDs = doneIDs(~cellfun(@isempty, doneIDs));
end
end


function append_history_record(historyFile, rawID, phase, cx, cy, boxSize, outputPrefix, inputDir, outputDir)
needHeader = ~exist(historyFile, 'file');

fid = fopen(historyFile, 'a');
if fid < 0
    return;
end

if needHeader
    fprintf(fid, 'type\traw_id\tphase\tcx\tcy\tbox_size\toutput_prefix\tinput_dir\toutput_dir\n');
end

fprintf(fid, 'crop\t%s\t%s\t%d\t%d\t%d\t%s\t%s\t%s\n', ...
    rawID, phase, cx, cy, boxSize, outputPrefix, inputDir, outputDir);

fclose(fid);
end


function mark_group_done(historyFile, rawID)
needHeader = ~exist(historyFile, 'file');

fid = fopen(historyFile, 'a');
if fid < 0
    return;
end

if needHeader
    fprintf(fid, 'type\traw_id\tphase\tcx\tcy\tbox_size\toutput_prefix\tinput_dir\toutput_dir\n');
end

fprintf(fid, 'done_group\t%s\t\t\t\t\t\t\t\n', rawID);
fclose(fid);
end


function tf = confirm_yes_no(promptText)
while true
    s = upper(strtrim(input(promptText, 's')));
    if any(strcmp(s, {'Y','N'}))
        tf = strcmp(s, 'Y');
        return;
    end
    disp('Please input Y or N.');
end
end


function n = next_cell_index(outputDir)
files1 = dir(fullfile(outputDir, '*_Tubulin.png'));
files2 = dir(fullfile(outputDir, '*_DNA.png'));
files = [files1; files2];
nums = [];
for i = 1:numel(files)
    tok = regexp(files(i).name, '^(\d+)_', 'tokens', 'once');
    if ~isempty(tok)
        nums(end+1) = str2double(tok{1});
    end
end
if isempty(nums)
    n = 1;
else
    n = max(nums) + 1;
end
end