"""This module implements CROTGate"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass

class CROTGate(QubitGate, DifferentiableUnitary, CachedClass):
    """
    The U2 single qubit gate.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        \\frac{\\sqrt{2}}{2} & -\\exp({i\\theta_1})\\frac{\\sqrt{2}}{2} \\\\
        \\exp({i\\theta_0})\\frac{\\sqrt{2}}{2} &
         \\exp({i(\\theta_0 + \\theta_1)})\\frac{\\sqrt{2}}{2} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _num_params = 2

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        theta = params[0]
        phi = params[1]

        return UnitaryMatrix(
            [
                [np.cos( theta / 2), -1j*(np.exp(-1j* phi))*np.sin(theta/2),0,0],
                [-1j*(np.exp(1j* phi))*np.sin(theta/2), np.cos( theta / 2),0,0],
                [0,0,np.cos( theta / -2), -1j*(np.exp(-1j* phi))*np.sin(theta/-2)],
                [0,0,-1j*(np.exp(1j* phi))*np.sin(theta/-2), np.cos( theta / -2)],

            ],
        )
        
    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)
        theta = params[0]
        phi = params[1]
        return np.array(
            [
            [
                [-1/2*np.sin( theta / 2), -1j*(np.exp(-1j* phi))*np.cos(theta/2)/2, 0, 0],
                [-1j*(np.exp(1j* phi))*np.cos(theta/2)/2, -1/2*np.sin( theta / 2), 0, 0],
                [0, 0, 1/2*np.sin( theta / -2), -1j*(np.exp(-1j* phi))*np.cos(theta/-2)/-2],
                [0, 0, -1j*(np.exp(1j* phi))*np.cos(theta/-2)/-2, np.cos( theta / -2)/2],

            ],
            [
                [0, -1*(np.exp(-1j* phi))*np.sin(theta/2), 0, 0],
                [(np.exp(1j* phi))*np.sin(theta/2), 0, 0, 0],
                [0, 0, 0, -1*(np.exp(-1j* phi))*np.sin(theta/-2)],
                [0, 0, (np.exp(1j* phi))*np.sin(theta/-2), 0],

            ],
            ],
        )
