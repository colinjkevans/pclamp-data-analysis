A few different kinds of traces, used to calculate different things.

1) current steps -- mostly used to define features of action potentials (APs), both single APs and AP trains
    - \# APs per amount of current injected, where APl is defined as having at least 40mV amplitude and has to overshoot 0mV
    - Rheobase: current step that first elicits at least one AP
    - Spike frequency adaptation: ratio of first to 10th interspike interval (ISI1/ISI10) and ratio of first to last interspike interval (ISI1/ISIn) for the first suprathreshold current injection to elicit sustained firing
    - Max steady state firing frequency: max mean firing frequency in response to current injection with no failures (AP amplitude at least 40mV and overshooting 0mV)
    - Max instantaneous firing frequency: inverse of smallest interspike interval in response to current injection as per above
    - AP threshold #1: for first spike obtained at suprathreshold current injection, the voltage at which first derivative (dV/dt) of the AP waveform reaches 10ms
    - AP threshold #2: for first spike obtained at suprathreshold current injection, the voltage at which second derivative (d2V/dt2) reaches 5% of maximal value
    - AP rise time: for first spike obtained at suprathreshold current injection, time from AP threshold to peak
    - AP amplitude: for first spike obtained at suprathreshold current injection, change in mV from AP threshold #1 to peak
    - AP half-width: for first spike obtained at suprathreshold current injection, width of the AP (in ms) at 1/2 maximal amplitude, using AP threshold #1 and AP amplitude
    - Input resistance #1: calculate using slope of the linear fit to the plot of the V-I relation from subthreshold current steps at/around resting potential

2) VC test
    - Input resistance #2: calculate using change in steady state current in response to small hyperpolarizing voltage step

3) current clamp gap free
    - resting potential: average membrane potential across the recording

4) voltage clamp gap free (deferred)
    - EPSC magnitude and frequency
    - IPSC magnitude and frequency

5) perforant path stimulation (deferred)
    - EPSC / IPSC magnitude and time to onset