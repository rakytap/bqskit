"""This module implements the SquanderSynthesisPass."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.passdata import PassData

import math 

from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.passes.synthesis.synthesis import SynthesisPass
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number
from squander import utils
from squander.gates import qgd_Circuit
from squander import N_Qubit_Decomposition_Tree_Search
from squander import N_Qubit_Decomposition_Tabu_Search

_logger = logging.getLogger(__name__)


class SquanderSynthesisPass(SynthesisPass):
    """
    A pass implementing the Squander synthesis algorithm.

    """

    def __init__(
        self, success_threshold : float = 1e-8,
        max_layer: int | None = None,
        squander_config: dict[str, Any] = {} 
    ) -> None:
        """
        Construct a search-based synthesis pass.

        Args:
            success_threshold (float): The distance threshold that
                determines successful termintation. Measured in cost
                described by the cost function. (Default: 1e-8)

            max_layer (int): The maximum number of layers to append without
                success before termination. If left as None it will default
                to unlimited. (Default: None)
                
            squander_config: 

        Raises:
            ValueError: If `max_depth` is nonpositive.
        """

        if not is_real_number(success_threshold):
            raise TypeError(
                'Expected real number for success_threshold'
                f', got {type(success_threshold)}',
            )

        if max_layer is not None and not is_integer(max_layer):
            raise TypeError(
                f'Expected max_layer to be an integer, got {type(max_layer)}.',
            )

        if max_layer is not None and max_layer <= 0:
            raise ValueError(
                f'Expected max_layer to be positive, got {int(max_layer)}.',
            )

        #TODO checking inputs and docstrings
        #TODO set cost function according to BQskit
        #TODO implement cost function variant from input config
        #TODO implement verbosity from config
        #TODO implement default values for squander config
        #TODO add optimizer engine field into squander config
        #TODO References:  Dr. Peter Rakyta ?????????


        self.success_threshold = success_threshold

        # cost calculator from BQSkit to verify squander sythesis
        self.bqskit_cost_calculator = HilbertSchmidtResidualsGenerator()

        # the maximum number of layers (CNOT gates) used in the tree search
        self.max_layer = max_layer

        self.squander_config = squander_config


        squander_config.setdefault("verbosity", -1)
        squander_config.setdefault("strategy", "Tabu_search")
        squander_config.setdefault("optimization_tolerance", 1e-8)
        squander_config.setdefault("Cost_Function_Variant",3)
        squander_config.setdefault("optimizer_engine",'BFGS')
        


     
    def transform_circuit_from_squander_to_bqskit(self,
        Squander_circuit,
        parameters,utry: UnitaryMatrix,)-> Circuit:
        '''A function to translate the circuit from squander to bqskit 
        
           Args:
               Squander_circuit: circuit made with squander.
               parameters: paramteres of the gates.
        
        '''
  
        
    #import all the gates
        from bqskit.ir.gates.constant.cx import CNOTGate
        from bqskit.ir.gates.parameterized.cry import CRYGate
        from bqskit.ir.gates.constant.cz import CZGate
        from bqskit.ir.gates.constant.ch import CHGate
        from bqskit.ir.gates.constant.sycamore import SycamoreGate
        from bqskit.ir.gates.parameterized.u3 import U3Gate 
        from bqskit.ir.gates.parameterized.rx import RXGate
        from bqskit.ir.gates.parameterized.ry import RYGate
        from bqskit.ir.gates.parameterized.rz import RZGate
        from bqskit.ir.gates.constant.x import XGate
        from bqskit.ir.gates.constant.y import YGate
        from bqskit.ir.gates.constant.z import ZGate
        from bqskit.ir.gates.constant.sx import SqrtXGate
        import squander
        
        Umtx = utry.numpy
        if self.squander_config["strategy"] == "Tree_search":
            cDecompose = N_Qubit_Decomposition_Tree_Search( Umtx.conj().T, config=self.squander_config, accelerator_num=0 )
        elif self.squander_config["strategy"] == "Tabu_search":
            cDecompose = N_Qubit_Decomposition_Tabu_Search( Umtx.conj().T, config=self.squander_config, accelerator_num=0 )
            
        qbit_num = cDecompose.get_Qbit_Num()
        circuit = Circuit(qbit_num)
        gates = Squander_circuit.get_Gates()

        # constructing quantum circuit
        for gate in gates:

            if isinstance( gate, squander.CNOT ):
                # adding CNOT gate to the quantum circuit
                control_qbit = qbit_num - gate.get_Control_Qbit()  - 1              
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1               
                circuit.append_gate(CNOTGate(), (control_qbit, target_qbit))
            
            elif isinstance( gate, squander.CRY ):
                # adding CNOT gate to the quantum circuit
                parameters_gate = gate.Extract_Parameters( parameters )
                control_qbit = qbit_num - gate.get_Control_Qbit() - 1                
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1               
                circuit.append_gate(CRYGate() ,(control_qbit, target_qbit), parameters_gate)
            
            elif isinstance( gate, squander.CZ ):
                # adding CZ gate to the quantum circuit
                control_qbit = qbit_num - gate.get_Control_Qbit() - 1               
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1                  
                circuit.append_gate(CZGate(), (control_qbit, target_qbit))

            elif isinstance( gate, squander.CH ):    
                # adding CH gate to the quantum circuit
                control_qbit = qbit_num - gate.get_Control_Qbit() - 1               
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1             
                circuit.append_gate(CZGate(), (control_qbit, target_qbit))

            elif isinstance( gate, squander.SYC ):
                # Sycamore gate
                control_qbit = qbit_num - gate.get_Control_Qbit() - 1               
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1            
                circuit.append_gate(SycamoreGate(), (control_qbit, target_qbit))

            elif isinstance( gate, squander.U3 ):
                # adding U3 gate to the quantum circuit
                parameters_gate = gate.Extract_Parameters( parameters )
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1      
                circuit.append_gate(U3Gate(),target_qbit, parameters_gate)   

            elif isinstance( gate, squander.RX ):
                # RX gate
                parameters_gate = gate.Extract_Parameters( parameters )
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1        
                circuit.append_gate(RXGate(),( target_qbit), parameters_gate)   
            
            elif isinstance( gate, squander.RY ):
                # RY gate
                parameters_gate = gate.Extract_Parameters( parameters )
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1   
                circuit.append_gate(RYGate(),(target_qbit), parameters_gate)

            elif isinstance( gate, squander.RZ ):
                # RZ gate
                parameters_gate = gate.Extract_Parameters( parameters )
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1    
                circuit.append_gate(RZGate(), (target_qbit), parameters_gate ) 
            
            elif isinstance( gate, squander.H ):
                # Hadamard gate
                circuit.h( gate.get_Target_Qbit() )    

            elif isinstance( gate, squander.X ):
                # X gate
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1      
                circuit.append_gate(XGate(), (target_qbit))

            elif isinstance( gate, squander.Y ):
                # Y gate
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1      
                circuit.append_gate(YGate(), (target_qbit))

            elif isinstance( gate, squander.Z ):
                # Z gate
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1      
                circuit.append_gate(ZGate(), (target_qbit))

            elif isinstance( gate, squander.SX ):
                # SX gate
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1      
                circuit.append_gate(SqrtXGate(), (target_qbit))

            elif isinstance( gate, squander.Circuit ):
                # Sub-circuit gate
                raise ValueError("Qiskit export of circuits with subcircuit is not supported. Use Circuit::get_Flat_Circuit prior of exporting circuit.")  
            
            else:
                print(gate)
                raise ValueError("Unsupported gate in the circuit export.")

  
        return(circuit)




    async def synthesize(
        self,
        utry: UnitaryMatrix,
        data: PassData,
    ) -> Circuit:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""
        # Initialize run-dependent options
        

        
        Umtx = utry.numpy
        qbit_num = math.floor(math.log2(Umtx.shape[0])) 

        

       
        if self.squander_config["strategy"] == "Tree_search":
            cDecompose = N_Qubit_Decomposition_Tree_Search( Umtx.conj().T, config=self.squander_config, accelerator_num=0 )
        elif self.squander_config["strategy"] == "Tabu_search":
            cDecompose = N_Qubit_Decomposition_Tabu_Search( Umtx.conj().T, config=self.squander_config, accelerator_num=0 )

            

        cDecompose.set_Verbose( self.squander_config["verbosity"] )
        cDecompose.set_Cost_Function_Variant(self.squander_config["Cost_Function_Variant"])

    

        # adding new layer to the decomposition until threshold
        cDecompose.set_Optimizer( self.squander_config["optimizer_engine"] )

        # starting the decomposition
        cDecompose.Start_Decomposition()
            

        squander_circuit = cDecompose.get_Circuit()
        parameters       = cDecompose.get_Optimized_Parameters()
   
        Circuit_squander = self.transform_circuit_from_squander_to_bqskit( squander_circuit, parameters,utry)          
        dist             = self.bqskit_cost_calculator.calc_cost(Circuit_squander, utry)  
        
        #print( 'Squander dist: ', str(dist) )
           

        _logger.debug("The error of the decomposition with SQUANDER is "  + str(dist))            
           
            
        if dist >  self.success_threshold:
            _logger.debug('the squander decomposition error is bigger than the succes_treshold, with the value of:', dist)                
        else: 
            _logger.debug('Successful synthesis with squander.')

        '''
        squander_gates = {}

        for gate in Circuit_squander.gate_set:
            squander_gates[str(gate)] = Circuit_squander.count(gate)
 
        #print('squander: qbit_num = ', qbit_num, ' dist = ', dist, squander_gates, "\n")
        '''
        
           
        return(Circuit_squander) 

            
         

        return layer_gen
