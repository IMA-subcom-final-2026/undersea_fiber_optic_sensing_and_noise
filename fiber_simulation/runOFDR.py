import numpy as np
import matplotlib.pyplot as plt

from getBackscat import Backscat
from getFiber import Fiber
from getProbe import linear_chirp, nonlinear_chirp, custom_probe
from getDisturbance import sinusoidal_phase, gaussian_burst_phase, damped_ringdown_phase, custom_phase
from getLinearChirpPhase import LinearChirpPhase
from getMatchedFilterPhase import MatchedFilterPhase


C0 = 299792458.0


def make_probe(config):
    if config['probe_type'] == 'linear_chirp':
        return lambda t: linear_chirp(t, config['sweep_rate'])

    if config['probe_type'] == 'nonlinear_chirp':
        return lambda t: nonlinear_chirp(t, config['duration'], config['sweep_span'], config['nonlinearity'])

    return lambda t: custom_probe(t, lambda u: config['custom_probe_function'](u, config['duration'], config['sweep_span']))


def make_phases(config, number_of_stretchers):
    phases = []
    for m in range(number_of_stretchers):
        amplitude = config['phase_amplitude'] * (1.0 + 0.1 * m)
        frequency = config['phase_frequency'] + config['frequency_spacing'] * m

        if config['phase_type'] == 'sinusoidal':
            phase = sinusoidal_phase(amplitude, frequency)
        elif config['phase_type'] == 'gaussian_burst':
            phase = gaussian_burst_phase(amplitude, frequency, config['duration'] / 2.0, config['burst_width'])
        elif config['phase_type'] == 'damped_ringdown':
            phase = damped_ringdown_phase(amplitude, frequency, config['duration'] / 4.0, config['decay_time'])
        else:
            phase = custom_phase(lambda t, m=m, amplitude=amplitude, frequency=frequency: config['custom_phase_function'](t, m, amplitude, frequency))

        phases.append(phase)
    return phases


def phase_frequency_limit(config, number_of_stretchers):
    highest = config['phase_frequency'] + config['frequency_spacing'] * (number_of_stretchers - 1)
    if config['phase_type'] == 'gaussian_burst':
        highest += 1.0 / (2.0 * np.pi * config['burst_width'])
    if config['phase_type'] == 'damped_ringdown':
        highest += 1.0 / (2.0 * np.pi * config['decay_time'])
    if config['phase_type'] == 'custom':
        highest = config['custom_phase_max_frequency']
    return highest


def expected_phase(fiber, time, channel_centers, boundaries, phases, mode):
    cumulative = []
    for i, center in enumerate(channel_centers):
        if mode == 'quasistatic':
            phase = fiber.quasistatic_phase(time, i, phases)
        else:
            phase = fiber.dynamic_phase(time, i, center - boundaries[i], phases)
        cumulative.append(0.5 * phase)
    return np.diff(np.asarray(cumulative), axis=0)


def linear_chirp_recovery(config, fiber, t, probe, phases, b_ref, b_dstb, boundaries):
    detector = LinearChirpPhase()
    detected_ref = detector.homodyne(b_ref, probe(t))
    detected_dstb = detector.homodyne(b_dstb, probe(t))
    distance, ref_spectrum = detector.distance_spectrum(detected_ref, 1.0 / config['sample_rate'], config['sweep_rate'], config['n_eff'])
    _, dstb_spectrum = detector.distance_spectrum(detected_dstb, 1.0 / config['sample_rate'], config['sweep_rate'], config['n_eff'])

    range_resolution = C0 / (2.0 * config['n_eff'] * config['sweep_span'])
    highest_phase_frequency = phase_frequency_limit(config, len(fiber.section_lengths) - 1)
    phase_sideband_distance = C0 * highest_phase_frequency / (2.0 * config['n_eff'] * config['sweep_rate'])
    window_length = min(max(config['spatial_window_length'], 4.0 * range_resolution, 4.0 * phase_sideband_distance), np.min(fiber.section_lengths) / 4.0)
    channel_centers = []
    ref_channels = []
    dstb_channels = []
    for i in range(len(fiber.section_lengths)):
        center = detector.strongest_channel_center(ref_spectrum, distance, boundaries[i], boundaries[i + 1], window_length)
        channel_centers.append(center)
        ref_channels.append(detector.isolate_channel(ref_spectrum, distance, center, window_length))
        dstb_channels.append(detector.isolate_channel(dstb_spectrum, distance, center, window_length))

    recovered_cumulative = []
    for ref_channel, dstb_channel in zip(ref_channels, dstb_channels):
        recovered_cumulative.append(detector.cumulative_phase(dstb_channel, ref_channel))

    recovered_local = np.diff(np.asarray(recovered_cumulative), axis=0)
    model_local = expected_phase(fiber, t, channel_centers, boundaries, phases, config['phase_mode'])
    inside = (distance >= 0.0) & (distance <= config['total_length'])
    order = np.argsort(distance[inside])
    return distance[inside][order], ref_spectrum[inside][order], t, np.asarray(channel_centers), recovered_local, model_local, 'linear FFT', None


def matched_filter_recovery(config, fiber, t, probe, phases, b_ref, b_dstb, boundaries):
    detector = MatchedFilterPhase()
    detected_ref = detector.homodyne(b_ref, probe(t))
    detected_dstb = detector.homodyne(b_dstb, probe(t))
    distance = np.linspace(0.0, config['total_length'], config['profile_points'])
    ref_profile = detector.matched_filter_profile(detected_ref, t, distance, probe, config['n_eff'])

    margin = min(config['spatial_window_length'], np.min(fiber.section_lengths) / 4.0)
    channel_centers = []
    for i in range(len(fiber.section_lengths)):
        channel_centers.append(detector.strongest_channel_center(ref_profile, distance, boundaries[i], boundaries[i + 1], margin))

    highest_phase_frequency = phase_frequency_limit(config, len(fiber.section_lengths) - 1)
    window_duration = min(config['matched_window_duration'], 0.2 / highest_phase_frequency)
    half_window = window_duration / 2.0
    time_centers = np.linspace(half_window, config['duration'] - half_window, config['phase_time_points'])
    ref_channels = detector.short_time_channels(detected_ref, t, channel_centers, probe, config['n_eff'], time_centers, window_duration)
    dstb_channels = detector.short_time_channels(detected_dstb, t, channel_centers, probe, config['n_eff'], time_centers, window_duration)
    recovered_cumulative = detector.cumulative_phase(dstb_channels, ref_channels)
    recovered_local = np.diff(recovered_cumulative, axis=0)
    model_local = expected_phase(fiber, time_centers, channel_centers, boundaries, phases, config['phase_mode'])
    return distance, ref_profile, time_centers, np.asarray(channel_centers), recovered_local, model_local, 'matched filter', window_duration


def run_experiment(config):
    t = np.arange(0.0, config['duration'], 1.0 / config['sample_rate'])
    probe = make_probe(config)
    number_of_stretchers = len(config['stretcher_positions'])
    phases = make_phases(config, number_of_stretchers)

    center_rng = np.random.default_rng(config['seed'])
    coeff_rng = np.random.default_rng(config['seed'] + 1)
    generator = Backscat(1, config['center_distribution'], 1, config['coefficient_distribution'], center_rng, coeff_rng)
    section_lengths, z_sections, a_sections = generator.make_fiber(config['total_length'], config['stretcher_positions'], config['scatterers_per_meter'], config['center_distribution'], config['coefficient_distribution'], config['min_scatterers_per_section'])

    attenuation = config['attenuation_db_per_km'] if config['use_attenuation'] else 0.0
    beta0 = 2.0 * np.pi * config['phase_index'] / config['wavelength']
    beta1 = config['n_eff'] / C0
    fiber = Fiber(section_lengths, z_sections, a_sections, beta0, beta1, attenuation)
    b_ref, b_dstb, ref_sections, dstb_sections = fiber.simulate(t, probe, phases, config['phase_mode'])

    method = config['detection_method']
    if config['probe_type'] != 'linear_chirp':
        method = 'matched_filter'
    elif method == 'auto':
        method = 'linear_fft'

    boundaries = np.r_[0.0, config['stretcher_positions'], config['total_length']]
    if method == 'linear_fft':
        recovery = linear_chirp_recovery(config, fiber, t, probe, phases, b_ref, b_dstb, boundaries)
    else:
        recovery = matched_filter_recovery(config, fiber, t, probe, phases, b_ref, b_dstb, boundaries)

    distance, profile, phase_time, channel_centers, recovered_local, model_local, detector_name, matched_window_used = recovery
    input_phase = np.asarray([phase(phase_time) for phase in phases])
    rmse = np.sqrt(np.mean((recovered_local - input_phase)**2, axis=1))
    detector_rmse = np.sqrt(np.mean((recovered_local - model_local)**2, axis=1))
    return {
        'section_lengths': section_lengths,
        'scatterers_per_section': np.asarray([len(z) for z in z_sections]),
        'stretcher_positions': config['stretcher_positions'],
        'distance': distance,
        'reference_profile': profile,
        'phase_time': phase_time,
        'channel_centers': channel_centers,
        'recovered_local': recovered_local,
        'input_phase': input_phase,
        'model_local': model_local,
        'rmse': rmse,
        'detector_rmse': detector_rmse,
        'detector_name': detector_name,
        'matched_window_used': matched_window_used,
        'reference_sections': ref_sections,
        'disturbed_sections': dstb_sections,
    }


def show_experiment(result, config):
    profile = np.abs(result['reference_profile'])
    profile_db = 20.0 * np.log10(np.maximum(profile / np.max(profile), 1e-7))
    attenuation = f"{config['attenuation_db_per_km']} dB/km" if config['use_attenuation'] else 'off'
    phase_detail = ''
    if config['phase_type'] == 'gaussian_burst':
        phase_detail = f", width={config['burst_width'] * 1e3:g} ms"
    if config['phase_type'] == 'damped_ringdown':
        phase_detail = f", decay={config['decay_time'] * 1e3:g} ms"
    matched_window = result['matched_window_used'] if result['matched_window_used'] is not None else config['matched_window_duration']
    subtitle = (f"L={config['total_length']} m, sections={np.round(result['section_lengths'], 3)} m, centers={config['center_distribution']}, coefficients={config['coefficient_distribution']}, target density={config['scatterers_per_meter']}/m, min/section={config['min_scatterers_per_section']}, seed={config['seed']}\n"
                f"probe={config['probe_type']}, detector={result['detector_name']}, fs={config['sample_rate'] / 1e6:g} MHz, T={config['duration'] * 1e3:g} ms, span={config['sweep_span'] / 1e9:.3g} GHz, nonlinear={config['nonlinearity']}\n"
                f"phase={config['phase_type']}, A={config['phase_amplitude']} rad, f0={config['phase_frequency']} Hz, df={config['frequency_spacing']} Hz{phase_detail}, mode={config['phase_mode']}\n"
                f"attenuation={attenuation}, n_phase={config['phase_index']}, n_group={config['n_eff']}, wavelength={config['wavelength'] * 1e9:g} nm, spatial window={config['spatial_window_length']} m, matched window={matched_window * 1e3:.3g} ms")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    fig.suptitle('OFDR phase recovery\n' + subtitle, fontsize=8.5)
    axes[0].plot(result['distance'], profile_db)
    for position in result['stretcher_positions']:
        axes[0].axvline(position, color='C3', linestyle='--', alpha=0.7)
    axes[0].plot(result['channel_centers'], np.full(result['channel_centers'].size, -4.0), 'kv')
    axes[0].set(xlabel='distance (m)', ylabel='reference profile (dB)', ylim=(-70, 3))

    for m in range(result['recovered_local'].shape[0]):
        axes[1].plot(result['phase_time'] * 1e3, result['input_phase'][m], color=f'C{m}', alpha=0.45, label=f'input g{m + 1}')
        axes[1].plot(result['phase_time'] * 1e3, result['recovered_local'][m], color=f'C{m}', linestyle='--', label=f'recovered g{m + 1}')
    axes[1].set(xlabel='time (ms)', ylabel='local phase (rad)')
    axes[1].legend(ncol=2)
    plt.show()

    print('section lengths (m):', np.round(result['section_lengths'], 3))
    print('scatterers per section:', result['scatterers_per_section'])
    print('stretcher positions (m):', np.round(result['stretcher_positions'], 3))
    print('selected channel centers (m):', np.round(result['channel_centers'], 3))
    print('g(t) reconstruction RMSE (rad):', np.round(result['rmse'], 5))
    if config['phase_mode'] == 'dynamic':
        print('detector-to-center-model RMSE (rad):', np.round(result['detector_rmse'], 5))
