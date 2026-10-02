# Propeller acoustic mesh resolution and tip Mach number

This note derives the acoustic mesh spacing for a propeller simulation using a permeable FW–H approach, assuming **16 mesh points per acoustic wavelength**. It includes the relationship to tip Mach, a worked example at 1000 RPM, and the distinction between acoustic and flow-field resolution.

## 1. Definitions and assumptions

| Symbol | Meaning | Units |
| --- | --- | --- |
| Δx_max | Maximum allowable characteristic mesh spacing under the PPW criterion | m |
| c | Speed of sound; assumed to be 343 m/s in the example | m/s |
| f_max | Highest acoustic frequency to resolve | Hz |
| N_PPW | Number of mesh points per acoustic wavelength; 16 here | — |
| B | Number of blades | — |
| n | Highest BPF harmonic to resolve | — |
| D | Propeller diameter | m |
| R | Propeller radius, D/2 | m |
| M_tip | Rotational blade-tip Mach number | — |
| Ω | Angular velocity | rad/s |

The wavelength calculation below assumes acoustic propagation in a stationary medium with uniform sound speed. A mean flow changes the wavelength according to propagation direction and local flow velocity; rotational tip Mach is not the local mean-flow Mach in the acoustic propagation region.

**Terminology:** This calculation gives a maximum allowable spacing, rather than a minimum edge length. Smaller cells are acceptable and may be necessary to resolve the flow and noise sources. On an unstructured mesh, characteristic spacing along the propagation direction matters; a single edge-length setting is only a practical proxy.

## 2. Blade passing frequency and target bandwidth

The blade passing frequency is:

$$f_{\mathrm{BPF}} = B\frac{\mathrm{RPM}}{60}$$

Plain text: **BPF = B × RPM / 60**

If the target is the nth BPF harmonic:

$$f_{\max} = n f_{\mathrm{BPF}} = nB\frac{\mathrm{RPM}}{60}$$

Plain text: **f_max = n × BPF = n × B × RPM / 60**

The harmonic number must follow the required frequency bandwidth. Three to five harmonics is not a universal aeroacoustic requirement. Broadband noise also needs a specified upper frequency; describing that frequency as a BPF multiple does not imply the noise is tonal.

## 3. Relate rotational speed to tip Mach

The rotational tip speed is:

$$V_{\mathrm{tip}} = \Omega R = \frac{2\pi R\,\mathrm{RPM}}{60} = \frac{\pi D\,\mathrm{RPM}}{60}$$

The rotational tip Mach number is:

$$M_{\mathrm{tip}} = \frac{V_{\mathrm{tip}}}{c} = \frac{\pi D\,\mathrm{RPM}}{60c}$$

Plain text: **M_tip = π × D × RPM / (60 × c)**

Rearranging:

$$\frac{\mathrm{RPM}}{60} = \frac{M_{\mathrm{tip}}c}{\pi D}$$

$$\mathrm{RPM} = \frac{60M_{\mathrm{tip}}c}{\pi D}$$

Therefore:

$$f_{\mathrm{BPF}} = \frac{BM_{\mathrm{tip}}c}{\pi D}$$

$$f_{\max} = \frac{nBM_{\mathrm{tip}}c}{\pi D}$$

These equations use rotational tip Mach. If a reported tip Mach includes axial or other relative velocity components, use the rotational component for the RPM relationship.

## 4. Shortest acoustic wavelength and mesh spacing

The shortest acoustic wavelength in the target band is:

$$\lambda_{\min} = \frac{c}{f_{\max}}$$

For N_PPW points per wavelength:

$$\Delta x \leq \Delta x_{\max} = \frac{\lambda_{\min}}{N_{\mathrm{PPW}}} = \frac{c}{N_{\mathrm{PPW}}f_{\max}}$$

With 16 PPW:

$$\boxed{\Delta x_{\max} = \frac{c}{16f_{\max}}}$$

Plain text: **Δx_max = c / (16 × f_max)**

Here, 16 PPW is an assumed design criterion. The actual resolution needed depends on the numerical scheme, dissipation, propagation distance, mesh quality, and acceptable error. Generic claims such as 20 PPW for all FEA/CAA methods or 6–10 PPW for all Lattice-Boltzmann methods should not be treated as universal requirements.

## 5. Express mesh spacing using tip Mach

Substituting the harmonic-based frequency into the PPW equation gives:

$$\Delta x_{\max} = \frac{c}{N_{\mathrm{PPW}}\left(nBM_{\mathrm{tip}}c/(\pi D)\right)}$$

Thus:

$$\boxed{\Delta x_{\max} = \frac{\pi D}{N_{\mathrm{PPW}}nBM_{\mathrm{tip}}} = \frac{2\pi R}{N_{\mathrm{PPW}}nBM_{\mathrm{tip}}}}$$

For 16 PPW:

$$\boxed{\Delta x_{\max} = \frac{\pi D}{16nBM_{\mathrm{tip}}}}$$

Plain text: **Δx_max = π × D / (16 × n × B × M_tip)**

For the BPF alone, n = 1:

$$\Delta x_{\max} = \frac{2\pi R}{16BM_{\mathrm{tip}}}$$

**Tip Mach alone cannot determine the mesh spacing.** This form additionally requires diameter, blade count, and the highest harmonic to resolve.

## 6. Does higher tip Mach allow larger cells?

**No, when D, B, and n are fixed.** Tip Mach appears in the denominator:

$$\Delta x_{\max} \propto \frac{1}{M_{\mathrm{tip}}}$$

Higher rotational tip Mach at fixed diameter means higher RPM, higher BPF, shorter wavelengths at the same harmonic number, and therefore smaller allowable spacing.

| M_tip | Spacing relative to the M_tip = 0.5 case |
| --- | --- |
| 0.5 | 1.000 |
| 0.7 | 0.714 |
| 0.9 | 0.556 |

For example, increasing tip Mach from 0.5 to 0.7 reduces the allowable spacing to 0.5/0.7 = 71.4% of its original value.

**If f_max is fixed in Hz instead**, the spacing is c/(16 × f_max). With the same sound speed and stationary-medium assumptions, changing tip Mach does not directly change this acoustic PPW requirement. It can still change the source physics and required flow-field resolution.

## 7. Worked example: 1000 RPM, two blades, 10 kHz

Given:

- RPM = 1000
- B = 2
- f_max = 10,000 Hz
- c = 343 m/s
- N_PPW = 16

### Blade passing frequency

$$f_{\mathrm{BPF}} = \frac{1000}{60}\times2 = 33.333\ldots\ \mathrm{Hz}$$

Plain text: **BPF = (1000 / 60) × 2 = 33.33 Hz**

### Equivalent harmonic number

$$n = \frac{10{,}000}{33.333\ldots} = 300$$

The upper frequency is the 300th BPF harmonic for this operating condition.

### Shortest wavelength

$$\lambda_{\min} = \frac{343}{10{,}000} = 0.0343\ \mathrm{m} = 34.3\ \mathrm{mm}$$

### Maximum allowable spacing

$$\Delta x_{\max} = \frac{343}{16\times10{,}000} = 0.00214375\ \mathrm{m}$$

$$\boxed{\Delta x_{\max} \approx 2.14\ \mathrm{mm}}$$

Plain text: **Δx_max = 343 / (16 × 10,000) = 0.00214375 m ≈ 2.14 mm**

For comparison, resolving only the BPF gives 643 mm, and resolving through the 10th BPF harmonic gives 64.3 mm. Those coarser values do not satisfy the 10 kHz target.

### Connection to M_tip = 0.7

At 1000 RPM and c = 343 m/s, rotational tip Mach 0.7 implies:

$$D = \frac{60M_{\mathrm{tip}}c}{\pi\,\mathrm{RPM}} = \frac{60\times0.7\times343}{\pi\times1000} \approx 4.586\ \mathrm{m}$$

Using this diameter and n = 300 in the tip-Mach formulation gives the same spacing:

$$\Delta x_{\max} = \frac{\pi D}{16\times300\times2\times0.7} \approx 2.14\ \mathrm{mm}$$

The frequency-based and tip-Mach-based expressions are equivalent when their inputs describe the same operating condition and target bandwidth.

## 8. Acoustic resolution versus convective and turbulence resolution

The acoustic wavelength above is c/f. A disturbance convecting at speed U_c has a characteristic streamwise length:

$$\lambda_{\mathrm{conv}} = \frac{U_c}{f}$$

If the same frequency and points-per-wavelength criterion are used for both:

$$\frac{\Delta x_{\mathrm{conv}}}{\Delta x_{\mathrm{acoustic}}} = \frac{U_c}{c}$$

Setting U_c equal to V_tip would give the scaling:

$$\Delta x_{\mathrm{conv}} \approx M_{\mathrm{tip}}\Delta x_{\mathrm{acoustic}}$$

However, **this is not a universal CFD mesh-design rule**. Blade speed is not automatically the convection speed of a wake or turbulent structure. LES, WMLES, and DES resolution also depends on boundary layers, shear layers, tip-vortex dimensions, turbulence scales, the numerical method, and the source mechanisms being modeled. A streamwise convective wavelength does not determine all cell dimensions.

## 9. Application to permeable FW–H

The CFD solution must transport the relevant fluctuations from the sources to the permeable integration surface with sufficient accuracy. Under the assumptions above, the 10 kHz acoustic PPW criterion gives approximately 2.14 mm characteristic spacing in that propagation region.

The criterion does not by itself establish adequate source resolution or accurate FW–H predictions. Surface placement, hydrodynamic disturbances crossing the surface, time-step and recording resolution, and mesh convergence also matter. The external observer field is evaluated by the FW–H formulation and does not require a CFD mesh extending to every observer.

## 10. Main formulas for copying

- **BPF = B × RPM / 60**
- **M_tip = π × D × RPM / (60 × c)**
- **f_max = n × BPF**
- **λ_min = c / f_max**
- **Δx_max = c / (N_PPW × f_max)**
- **Δx_max = π × D / (N_PPW × n × B × M_tip)**
- **For 16 PPW and 10 kHz at c = 343 m/s: Δx_max ≈ 2.14 mm.**

This note consolidates the supplied discussion and its qualifications. The numbered citation placeholders in the original pasted text had no accompanying source list and have therefore been omitted.
