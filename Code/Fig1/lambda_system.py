import numpy as np

class lambda_system:
  def P1(t, t0, d1, d2, g1, g2, in1, in2, in3):
    t = t-t0
    zeta = (-1 + np.sqrt(-3+0j))/2
    p = (d1+d2)**2 - 3*(d1*d2 - np.abs(g1)**2-np.abs(g2)**2) + 0j
    q = 2*(d1 + d2)**3 - \
        9*(d1 + d2)*(d1*d2 - np.abs(g1)**2 - np.abs(g2)**2) - \
        27*(d1*np.abs(g2)**2 + d2*np.abs(g1)**2) + 0j
    F = ((q-np.sqrt(q**2-4*p**3))/2)**(1/3)
    x = np.array([-(d1+d2 + F*zeta**i + p/(F*zeta**i))/3 for i in range(3)])

    numer = in1*(x**2 + d2*x - np.abs(g2)**2) - \
            in2*g1*(x + d2) + \
            in3*g1*g2
    denom = 3*x**2 + 2*(d1 + d2)*x + d1*d2 - np.abs(g1)**2 - np.abs(g2)**2

    coeff = numer/denom

    psi1 = 0.0
    for i in range(3):
      psi1 += coeff[i]*np.exp(-1j*x[i]*t)

    p1 = psi1 * np.conj(psi1)
    return p1

  def P2(t, t0, d1, d2, g1, g2, in1, in2, in3):
    t = t-t0
    zeta = (-1 + np.sqrt(-3+0j))/2
    p = (d1+d2)**2 - 3*(d1*d2 - np.abs(g1)**2-np.abs(g2)**2) + 0j
    q = 2*(d1 + d2)**3 - \
        9*(d1 + d2)*(d1*d2 - np.abs(g1)**2 - np.abs(g2)**2) - \
        27*(d1*np.abs(g2)**2 + d2*np.abs(g1)**2) + 0j
    F = ((q-np.sqrt(q**2-4*p**3))/2)**(1/3)
    x = np.array([-(d1+d2 + F*zeta**i + p/(F*zeta**i))/3 for i in range(3)])

    numer = -in1*np.conj(g1)*(x + d2) + \
            in2*(x**2 + (d1 + d2)*x + d1*d2) - \
            in3*g2*(x + d1)
    denom = 3*x**2 + 2*(d1 + d2)*x + d1*d2 - np.abs(g1)**2 - np.abs(g2)**2

    coeff = numer/denom

    psi2 = 0.0
    for i in range(3):
      psi2 += coeff[i]*np.exp(-1j*x[i]*t)

    p2 = psi2 * np.conj(psi2)
    return p2

  def P3(t, t0, d1, d2, g1, g2, in1, in2, in3):
    t = t-t0
    zeta = (-1 + np.sqrt(-3+0j))/2
    p = (d1+d2)**2 - 3*(d1*d2 - np.abs(g1)**2-np.abs(g2)**2) + 0j
    q = 2*(d1 + d2)**3 - \
        9*(d1 + d2)*(d1*d2 - np.abs(g1)**2 - np.abs(g2)**2) - \
        27*(d1*np.abs(g2)**2 + d2*np.abs(g1)**2) + 0j
    F = ((q-np.sqrt(q**2-4*p**3))/2)**(1/3)
    x = np.array([-(d1+d2 + F*zeta**i + p/(F*zeta**i))/3 for i in range(3)])

    numer = in1*np.conj(g1)*np.conj(g2) - \
            in2*np.conj(g2)*(x + d1) + \
            in3*(x**2 + d1*x - np.abs(g1)**2)
    denom = 3*x**2 + 2*(d1 + d2)*x + d1*d2 - np.abs(g1)**2 - np.abs(g2)**2

    coeff = numer/denom

    psi3 = 0.0
    for i in range(3):
      psi3 += coeff[i]*np.exp(-1j*x[i]*t)

    p3 = psi3 * np.conj(psi3)
    return p3


class UCCZS:
  def P1(t, t0, Delta, g, phi, in1, in2, in3):
    t=t-t0
    p = 2 * g**2 + Delta**2
    Omega = np.abs(np.sqrt(p))

    p1 = in1 * np.conj(in1) * (g**2 + Delta**2 + g**2 * np.cos(Omega * t))**2
    p1 += in2 * np.conj(in2) * 4 * g**2 * (g**2 + Delta**2 + g**2 * np.cos(Omega * t)) * np.sin(Omega * t / 2)**2
    p1 += in3 * np.conj(in3) * g**4 * (np.cos(Omega * t) - 1)**2
    p1 += - np.real(in1 * np.conj(in2) * (Delta * np.sin(Omega * t / 2) - 1j * Omega * np.cos(Omega * t / 2))) * 4 * g * (g**2 + Delta**2 + g**2 * np.cos(Omega * t)) * np.sin(Omega * t / 2)
    p1 += - np.real(in1 * np.conj(in3) * np.exp(-1j * phi) * (g**2 + (g**2 + Delta**2) * np.cos(Omega * t) + 1j * Delta * Omega * np.sin(Omega * t))) * 2 * g**2 * (np.cos(Omega * t) - 1)
    p1 += np.real(in2 * np.conj(in3) * np.exp(-1j * phi) * (Delta * (np.cos(Omega * t) - 1) + 1j * Omega * np.sin(Omega * t))) * 2 * g**3 * (np.cos(Omega * t) - 1)

    p1 = p1 / p**2
    return p1

  def P2(t, t0, Delta, g, phi, in1, in2, in3):
    t = t-t0
    p = 2 * g**2 + Delta**2
    Omega = np.abs(np.sqrt(p))

    p2 = in1 * np.conj(in1) * 4 * g**2 * (g**2 + Delta**2 + g**2 * np.cos(Omega * t)) * np.sin(Omega * t / 2)**2
    p2 += in2 * np.conj(in2) * (Delta**2 + 2 * g**2 * np.cos(Omega * t))**2
    p2 += in3 * np.conj(in3) * 4 * g**2 * (g**2 + Delta**2 + g**2 * np.cos(Omega * t)) * np.sin(Omega * t / 2)**2
    p2 += np.real(in1 * np.conj(in2) * (Delta * np.sin(Omega * t / 2) - 1j * Omega * np.cos(Omega * t / 2))) * 4 * g * (Delta**2 + 2 * g**2 * np.cos(Omega * t)) * np.sin(Omega * t / 2)
    p2 += - np.real(in1 * np.conj(in3) * np.exp(-1j * phi) * (g**2 + (g**2 + Delta**2) * np.cos(Omega * t) + 1j * Delta * Omega * np.sin(Omega * t))) * 8 * g**2 * np.sin(Omega * t / 2)**2
    p2 += - np.real(in2 * np.conj(in3) * np.exp(-1j * phi) * (Delta * (np.cos(Omega * t) - 1) + 1j * Omega * np.sin(Omega * t))) * 2 * g * (Delta**2 + 2 * g**2 * np.cos(Omega * t))

    p2 = p2 / p**2
    return p2

  def P3(t, t0, Delta, g, phi, in1, in2, in3):
    t = t-t0
    p = 2 * g**2 + Delta**2
    Omega = np.abs(np.sqrt(p))

    p3 = in1 * np.conj(in1) * g**4 * (np.cos(Omega * t) - 1)**2
    p3 += in2 * np.conj(in2) * 4 * g**2 * (g**2 + Delta**2 + g**2 * np.cos(Omega * t)) * np.sin(Omega * t / 2)**2 
    p3 += in3 * np.conj(in3) * (g**2 + Delta**2 + g**2 * np.cos(Omega * t))**2
    p3 += np.real(in1 * np.conj(in2) * (Delta * (np.cos(Omega * t) - 1) + 1j * Omega * np.sin(Omega * t))) * 2 * g**3 * (np.cos(Omega * t) - 1)
    p3 += - np.real(in1 * np.conj(in3) * np.exp(-1j * phi) * (g**2 + (g**2 + Delta**2) * np.cos(Omega * t) + 1j * Delta * Omega * np.sin(Omega * t))) * 2 * g**2 * (np.cos(Omega * t) - 1)
    p3 += - np.real(in2 * np.conj(in3) * np.exp(-1j * phi) * (Delta * np.sin(Omega * t / 2) - 1j * Omega * np.cos(Omega * t / 2))) * 4 * g * (g**2 + Delta**2 + g**2 * np.cos(Omega * t)) * np.sin(Omega * t / 2)

    p3 = p3 / p**2
    return p3