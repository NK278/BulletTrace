% FDTD Simulation in 2D with Layered Biological Tissues at 60 GHz
% Includes calculation of reflected Ez at five points by subtracting
% reference (incident-only) Ez from full-model Ez, and FFT of reflected Ez

clear all; close all; clc;
format long e;

%% Constants and Grid Parameters
c        = 3e8;                      % Speed of light (m/s)
f0       = 60e9;                     % Operating frequency (60 GHz)
lambda   = c / f0;                   % Wavelength (m)
delx_m   = lambda / 10;              % Spatial step (1/10th wavelength)
delt_sec = delx_m / (20 * c);        % Time step from CFL condition
ISTEPS   = 100;
JSTEPS   = 100;
NSTEPS   = 864;

xaxis_m = (0:ISTEPS-1) * delx_m;
yaxis_m = (0:JSTEPS-1) * delx_m;

%% Define five observation points (I,J) indices
pts = [ ...
    20, 10; ...
    20, 30; ...
    20, 50; ...
    20, 70; ...
    20, 90  ...
];
nPts = size(pts,1);

%% Precompute time vector and source profile (used by both runs)
t       = (0:NSTEPS-1) * delt_sec;
t0      = 4 * delt_sec;
bw      = 4e9;
sigma_t = sqrt(2) / (2 * pi * bw);
source  = exp(-((t - t0).^2) / (2 * sigma_t.^2)) .* cos(2 * pi * f0 * t);
ic      = round(5e-3 / delx_m) + 1;  % Line-source row index

epsilon_0 = 8.854e-12;
mu_0      = 4*pi*1e-7;

%% ------------------------------
%% Reference Simulation (homogeneous air)
%% ------------------------------

% Uniform epsilon_r and sigma for "air" (incident-only run)
epsilon_r_ref = ones(ISTEPS, JSTEPS);
sigma_ref     = zeros(ISTEPS, JSTEPS);

% Compute update coefficients for reference run
Zimp_ref        = 0.5 * 377 ./ epsilon_r_ref;
Yimp_ref        = 0.5 / 377 * ones(ISTEPS, JSTEPS);
sigma_star_ref  = sigma_ref .* (delt_sec ./ (2 * epsilon_0 .* epsilon_r_ref));
update_factor_ref = 1 ./ (1 + sigma_star_ref);

% Compute PML parameters for reference domain
[Abs_EbyH_ref, Abs_EbyE_ref, Abs_HbyE1_ref, Abs_HbyE2_ref, Abs_HbyH_ref] = ...
    calc_PML2(ISTEPS, JSTEPS, 12);

% Allocate field arrays for reference run
Ez_ref   = zeros(ISTEPS, JSTEPS, NSTEPS);
Hx_ref   = zeros(ISTEPS, JSTEPS, NSTEPS);
Hy_ref   = zeros(ISTEPS, JSTEPS, NSTEPS);
IHx_ref  = zeros(ISTEPS-1, JSTEPS-1);
IHy_ref  = zeros(ISTEPS-1, JSTEPS-1);

% Allocate array to store incident Ez at the five points
Ez_inc = zeros(nPts, NSTEPS);

% Time-stepping loop for reference run
for n = 2:NSTEPS
    % Update Ez in homogeneous air
    Ez_ref(2:ISTEPS, 2:JSTEPS, n) = ...
        update_factor_ref(2:ISTEPS, 2:JSTEPS) .* ...
        ( Ez_ref(2:ISTEPS, 2:JSTEPS, n-1) .* Abs_EbyE_ref + ...
          Abs_EbyH_ref .* Zimp_ref(2:ISTEPS, 2:JSTEPS) .* ...
           ( Hy_ref(2:ISTEPS, 2:JSTEPS, n-1) - Hy_ref(1:ISTEPS-1, 2:JSTEPS, n-1) ...
           - Hx_ref(2:ISTEPS, 2:JSTEPS, n-1) + Hx_ref(2:ISTEPS, 1:JSTEPS-1, n-1) ) );
    
    % Inject line source (same as full model)
    Ez_ref(ic, :, n) = Ez_ref(ic, :, n) + source(n);
    
    % Enforce PEC boundaries
    Ez_ref(:, [1, JSTEPS], n) = 0;
    Ez_ref([1, ISTEPS], :, n) = 0;

    % Update Hx_ref with PML
    Ez_diff_x_ref = Ez_ref(1:ISTEPS-1, 2:JSTEPS, n) - Ez_ref(1:ISTEPS-1, 1:JSTEPS-1, n);
    IHx_ref = IHx_ref + Abs_HbyE1_ref .* Ez_diff_x_ref;
    Hx_ref(1:ISTEPS-1, 1:JSTEPS-1, n) = ...
      Abs_HbyH_ref .* Hx_ref(1:ISTEPS-1, 1:JSTEPS-1, n-1) - ...
      Abs_HbyE2_ref .* Yimp_ref(1:ISTEPS-1, 1:JSTEPS-1) .* ( IHx_ref + Ez_diff_x_ref );

    % Update Hy_ref with PML
    Ez_diff_y_ref = Ez_ref(2:ISTEPS, 1:JSTEPS-1, n) - Ez_ref(1:ISTEPS-1, 1:JSTEPS-1, n);
    IHy_ref = IHy_ref + Abs_HbyE1_ref .* Ez_diff_y_ref;
    Hy_ref(1:ISTEPS-1, 1:JSTEPS-1, n) = ...
      Abs_HbyH_ref .* Hy_ref(1:ISTEPS-1, 1:JSTEPS-1, n-1) + ...
      Abs_HbyE2_ref .* Yimp_ref(1:ISTEPS-1, 1:JSTEPS-1) .* ( IHy_ref + Ez_diff_y_ref );

    % Record Ez_inc at the five points
    for p = 1:nPts
        i0 = pts(p,1);  j0 = pts(p,2);
        Ez_inc(p, n) = Ez_ref(i0, j0, n);
    end
end

%% ------------------------------
%% Full Simulation (layered chest + inclusions)
%% ------------------------------

%% Tissue Properties @ 60 GHz
eps_r_map = struct( ...
    'skin',   7.9753   , 'sigma_skin',   36.397, ...
    'fat',    3.1324   , 'sigma_fat',    2.8152, ...
    'muscle', 12.856   , 'sigma_muscle', 52.826, ...
    'bone',   3.8103   , 'sigma_bone',   7.2058, ...
    'blood',  9.5199   , 'sigma_blood',  37.34 , ...
    'nerve',  8.1698   , 'sigma_nerve',  26.481, ...
    'air',    1        , 'sigma_air',    0     );

% Initialize epsilon_r and sigma arrays to air
epsilon_r = ones(ISTEPS, JSTEPS) * eps_r_map.air;
sigma     = zeros(ISTEPS, JSTEPS);

% Layer thicknesses (m)
skin_thick   = 1.5e-3;
fat_thick    = 2e-3;
muscle_thick = 15e-3;
bone_thick   = 17e-3;

depths    = [skin_thick, fat_thick, muscle_thick, bone_thick];
cum_depth = cumsum([0, depths]);

x_start_flesh = 0.015;
I_start = round(x_start_flesh / delx_m) + 1;

% Assign layered ε_r and σ from I_start onward
for I = I_start : ISTEPS
    x_rel = (I - 1) * delx_m - x_start_flesh;
    if x_rel < cum_depth(2)
        epsilon_r(I, :) = eps_r_map.skin;
        sigma(I, :)     = eps_r_map.sigma_skin;
    elseif x_rel < cum_depth(3)
        epsilon_r(I, :) = eps_r_map.fat;
        sigma(I, :)     = eps_r_map.sigma_fat;
    elseif x_rel < cum_depth(4)
        epsilon_r(I, :) = eps_r_map.muscle;
        sigma(I, :)     = eps_r_map.sigma_muscle;
    elseif x_rel < cum_depth(5)
        epsilon_r(I, :) = eps_r_map.bone;
        sigma(I, :)     = eps_r_map.sigma_bone;
    end
end

%% Inclusions: Veins and Nerve
vein_x = 2.3e-2;
I_vein = round(vein_x / delx_m) + 1;
y_veins = [10e-3, 20e-3, 30e-3];
j_veins = round(y_veins / delx_m) + 1;

for k = 1:length(j_veins)
    jc = j_veins(k);
    for ii = max(1, I_vein-1) : min(ISTEPS, I_vein+1)
        for jj = max(1, jc-1) : min(JSTEPS, jc+1)
            epsilon_r(ii, jj) = eps_r_map.blood;
            sigma(ii, jj)     = eps_r_map.sigma_blood;
        end
    end
end

I_nerve = round(2.6e-2 / delx_m) + 1;
j_nerve = round(40e-3 / delx_m) + 1;

for ii = max(1, I_nerve-1) : min(ISTEPS, I_nerve+1)
    for jj = max(1, j_nerve-1) : min(JSTEPS, j_nerve+1)
        epsilon_r(ii, jj) = eps_r_map.nerve;
        sigma(ii, jj)     = eps_r_map.sigma_nerve;
    end
end

% A small high-conductivity block (e.g. metal inclusion)
mx1=70; mx2=75; my1=30; my2=40;
sigma(mx1:mx2, my1:my2)     = 1e6;
epsilon_r(mx1:mx2, my1:my2) = 1;

%% Build a discrete 'tissue-label' map (0–6)
tissue = zeros(ISTEPS, JSTEPS);

for I = I_start : ISTEPS
    x_rel = (I - 1) * delx_m - x_start_flesh;
    if x_rel < cum_depth(2)
        tissue(I, :) = 1;  % skin
    elseif x_rel < cum_depth(3)
        tissue(I, :) = 2;  % fat
    elseif x_rel < cum_depth(4)
        tissue(I, :) = 3;  % muscle
    elseif x_rel < cum_depth(5)
        tissue(I, :) = 4;  % bone
    end
end

for k = 1:length(j_veins)
    jc = j_veins(k);
    for ii = max(1, I_vein-1) : min(ISTEPS, I_vein+1)
        for jj = max(1, jc-1) : min(JSTEPS, jc+1)
            tissue(ii, jj) = 5;  % blood
        end
    end
end

for ii = max(1, I_nerve-1) : min(ISTEPS, I_nerve+1)
    for jj = max(1, j_nerve-1) : min(JSTEPS, j_nerve+1)
        tissue(ii, jj) = 6;  % nerve
    end
end

%% Plot the static “Layered biceps model”
figure(1);
imagesc(xaxis_m, yaxis_m, tissue');
axis xy;
xlabel('X (m)');
ylabel('Y (m)');
title('Layered chest model – colour‐coded tissues & line source');

cmap = [ ...
    240 240 240;  % air
    244 202 133;  % skin
    255 230 128;  % fat
    255 153 153;  % muscle
    176 112  48;  % bone
    204   0   0;  % blood
      0 204 204] / 255;  % nerve
colormap(cmap);

cb = colorbar;
set(cb, 'Ticks', 0:6, 'TickLabels', {'air','skin','fat','muscle','bone','blood','nerve'});
ylabel(cb, 'Tissue type');

hold on;
xline(5e-3, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Line source (5 mm)');
legend('show', 'Location', 'northwest');
axis tight;

%% Material-Dependent Parameters for full run
Zimp_full        = 0.5 * 377 ./ epsilon_r;
Yimp_full        = 0.5 / 377 * ones(ISTEPS, JSTEPS);
sigma_star_full  = sigma .* (delt_sec ./ (2 * epsilon_0 .* epsilon_r));
update_factor_full = 1 ./ (1 + sigma_star_full);

% Compute PML for full domain
[Abs_EbyH_full, Abs_EbyE_full, Abs_HbyE1_full, Abs_HbyE2_full, Abs_HbyH_full] = ...
    calc_PML2(ISTEPS, JSTEPS, 12);

% Allocate field arrays for full run
Ez_full   = zeros(ISTEPS, JSTEPS, NSTEPS);
Hx_full   = zeros(ISTEPS, JSTEPS, NSTEPS);
Hy_full   = zeros(ISTEPS, JSTEPS, NSTEPS);
IHx_full  = zeros(ISTEPS-1, JSTEPS-1);
IHy_full  = zeros(ISTEPS-1, JSTEPS-1);

% Allocate array to store total Ez at the five points
Ez_tot = zeros(nPts, NSTEPS);

% Time-stepping loop for full simulation
figure;
for n = 2:NSTEPS
    % Update Ez in layered model
    Ez_full(2:ISTEPS, 2:JSTEPS, n) = ...
        update_factor_full(2:ISTEPS, 2:JSTEPS) .* ...
        ( Ez_full(2:ISTEPS, 2:JSTEPS, n-1) .* Abs_EbyE_full + ...
          Abs_EbyH_full .* Zimp_full(2:ISTEPS, 2:JSTEPS) .* ...
           ( Hy_full(2:ISTEPS, 2:JSTEPS, n-1) - Hy_full(1:ISTEPS-1, 2:JSTEPS, n-1) ...
           - Hx_full(2:ISTEPS, 2:JSTEPS, n-1) + Hx_full(2:ISTEPS, 1:JSTEPS-1, n-1) ) );
    
    % Inject line source
    Ez_full(ic, :, n) = Ez_full(ic, :, n) + source(n);
    
    % Enforce PEC boundaries
    Ez_full(:, [1, JSTEPS], n) = 0;
    Ez_full([1, ISTEPS], :, n) = 0;

    % Update Hx_full with PML
    Ez_diff_x_full = Ez_full(1:ISTEPS-1, 2:JSTEPS, n) - Ez_full(1:ISTEPS-1, 1:JSTEPS-1, n);
    IHx_full = IHx_full + Abs_HbyE1_full .* Ez_diff_x_full;
    Hx_full(1:ISTEPS-1, 1:JSTEPS-1, n) = ...
      Abs_HbyH_full .* Hx_full(1:ISTEPS-1, 1:JSTEPS-1, n-1) - ...
      Abs_HbyE2_full .* Yimp_full(1:ISTEPS-1, 1:JSTEPS-1) .* ( IHx_full + Ez_diff_x_full );

    % Update Hy_full with PML
    Ez_diff_y_full = Ez_full(2:ISTEPS, 1:JSTEPS-1, n) - Ez_full(1:ISTEPS-1, 1:JSTEPS-1, n);
    IHy_full = IHy_full + Abs_HbyE1_full .* Ez_diff_y_full;
    Hy_full(1:ISTEPS-1, 1:JSTEPS-1, n) = ...
      Abs_HbyH_full .* Hy_full(1:ISTEPS-1, 1:JSTEPS-1, n-1) + ...
      Abs_HbyE2_full .* Yimp_full(1:ISTEPS-1, 1:JSTEPS-1) .* ( IHy_full + Ez_diff_y_full );

    % Record Ez_tot at the five points
    for p = 1:nPts
        i0 = pts(p,1);  j0 = pts(p,2);
        Ez_tot(p, n) = Ez_full(i0, j0, n);
    end

    % Real-time plotting of Ez_full magnitude (dB)
    Ez_dB = 20 * log10(abs(Ez_full(:,:,n))');
    imagesc(xaxis_m, yaxis_m, Ez_dB, [-50 0]);
    colorbar;
    set(gca, 'YDir', 'normal');
    xlabel('X (m)');
    ylabel('Y (m)');
    title(['Ez Field Magnitude (dB) at time step ', num2str(n)]);
    drawnow;
end

%% ------------------------------
%% Compute Reflected Field at the Five Points
%% ------------------------------
% Subtract incident-only Ez from total Ez
Ez_refl = Ez_tot - Ez_inc;  % size [5 × NSTEPS]

%% Plot Reflected Ez Time Histories
time_ns = (0:NSTEPS-1) * delt_sec * 1e9;  % convert to nanoseconds
figure;
for p = 1:nPts
    subplot(nPts,1,p);
    plot(time_ns, squeeze(Ez_refl(p,:)));
    ylabel(['Ez_{refl} @ (', num2str(pts(p,1)), ',', num2str(pts(p,2)), ')']);
    if p == nPts
        xlabel('Time (ns)');
    end
    grid on;
end
sgtitle('Reflected Ez at the Five Observation Points');

%% ------------------------------
%% Compute and Plot FFT of Reflected Ez
%% ------------------------------
fs = 1 / delt_sec;                    % Sampling frequency (Hz)
NFFT = NSTEPS;                        % Use full length
f_axis = (0:(NFFT/2-1)) * (fs / NFFT) / 1e9;  % Frequency axis in GHz (one-sided)

figure;
for p = 1:nPts
    E_fft = fft(Ez_refl(p, :), NFFT);       % FFT of reflected Ez at point p
    E_mag = abs(E_fft(1:NFFT/2));           % One-sided magnitude
    
    subplot(nPts,1,p);
    plot(f_axis, E_mag);
    xlabel('Frequency (GHz)');
    ylabel(['|FFT\{Ez_{refl}\}| @ (', num2str(pts(p,1)), ',', num2str(pts(p,2)), ')']);
    grid on;
    xlim([0, fs/2/1e9]);  % 0 to Nyquist in GHz
end
sgtitle('One-Sided FFT Magnitude of Reflected Ez');


filename = 'FDTD_sim_data6.mat';
save(filename, ...
     'Ez_refl', ...
     'epsilon_r', 'sigma', ...
     'xaxis_m', 'yaxis_m');
fprintf('Saved all data to %s\n',filename);