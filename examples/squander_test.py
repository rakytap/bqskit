from bqskit.ir import Circuit





circuit_name = 'tutorials/heisenberg-16-20' #9symml_195'


bqskit_circuit_original = Circuit.from_file( circuit_name + '.qasm')


from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.passes import QuickPartitioner, ClusteringPartitioner, GreedyPartitioner, ScanPartitioner
from bqskit.passes import ForEachBlockPass
from bqskit.passes import QSearchSynthesisPass, LEAPSynthesisPass, QFASTDecompositionPass, QPredictDecompositionPass
from bqskit.passes import ScanningGateRemovalPass
from bqskit.passes import UnfoldPass
from bqskit.passes import SquanderSynthesisPass
from bqskit.qis.unitary import Unitary
from qiskit import transpile
import numpy as np
import time
import pickle 
# import


# define the largest partition in circuit
largest_partition = 3

print("this partition is with ",largest_partition,"qubits on ",circuit_name,"circuit",)

start_squander = time.time()


###########################################################################
# SQUANDER synthesis
 
 
config = { 'max_outer_iterations': 1, 
                'agent_lifetime':1000,
                'max_inner_iterations_agent': 100000,
                'convergence_length': 10,
                'max_inner_iterations_compression': 100000,
                'max_inner_iterations' : 10000,
                'max_inner_iterations_final': 10000, 		
                'Randomized_Radius': 0.3, 
                'randomized_adaptive_layers': 1,
                'agent_num': 256,
                'parallel': 0,
                'batch_size': 4,
                'optimization_tolerance_agent': 1e-8} #1e-2
                

workflow = [
    ScanPartitioner( largest_partition ), 
    ForEachBlockPass([SquanderSynthesisPass(squander_config=config, optimizer_engine="BFGS" ), ScanningGateRemovalPass()]), 
    UnfoldPass(),
]
print("\n the original gates are: \n")

    
original_gates = []

for gate in bqskit_circuit_original.gate_set:
    case_original = {f"{gate}count:": bqskit_circuit_original.count(gate)}
    original_gates.append(case_original)
    
print(original_gates, "\n")

with open("original_gates.pickle", "wb") as file:
    pickle.dump(original_gates, file, pickle.HIGHEST_PROTOCOL)


# Finally, we construct a compiler and submit the task
#with Compiler(num_workers=1) as compiler:
with Compiler() as compiler:
    synthesized_circuit_squander = compiler.compile(bqskit_circuit_original, workflow)


Circuit.save(synthesized_circuit_squander, circuit_name + '_squander.qasm')




print("\n the gates with squander :")


squander_gates = []

for gate in synthesized_circuit_squander.gate_set:
    case_squander = {f"{gate}count:":  synthesized_circuit_squander.count(gate)}
    squander_gates.append(case_squander)
 
end_squander = time.time()
time_squander = {"the execution time with squander:": end_squander-start_squander}
squander_gates.append(time_squander)
print(squander_gates, "\n")

with open("squander_gates.pickle", "wb") as file:
    pickle.dump(squander_gates, file, pickle.HIGHEST_PROTOCOL)
    
    




###########################################################################
# QSearch synthesis

start_qsearch = time.time()


workflow = [
    QuickPartitioner( largest_partition ), 
    ForEachBlockPass([QSearchSynthesisPass(), ScanningGateRemovalPass()]), 
    UnfoldPass(),
]

# Finally, we construct a compiler and submit the task
with Compiler() as compiler:
    synthesized_circuit_qsearch = compiler.compile(bqskit_circuit_original, workflow)


# save the circuit is qasm format
Circuit.save(synthesized_circuit_qsearch, circuit_name + '_qsearch.qasm')



print("\n the gates with qsearch :")

qsearch_gates = []

for gate in synthesized_circuit_qsearch.gate_set:
    case_qsearch = {f"{gate}count:":  synthesized_circuit_qsearch.count(gate)}   
    qsearch_gates.append(case_qsearch)
 
end_qsearch = time.time()
time_qsearch = {"the execution time with qsearch:": end_qsearch-start_qsearch}
qsearch_gates.append(time_qsearch)
print(qsearch_gates, "\n")


with open("qsearch_gates.pickle", "wb") as file:
    pickle.dump(qsearch_gates, file, pickle.HIGHEST_PROTOCOL)






##############################################################################
#################### Test the generated circuits #############################

import qiskit
qiskit_version = qiskit.version.get_version_info()

from qiskit import QuantumCircuit
import qiskit_aer as Aer    
    
if qiskit_version[0] == '1':
    from qiskit import transpile
else :
    from qiskit import execute




# load the circuit from QASM format
qc_original = QuantumCircuit.from_qasm_file( circuit_name +  '.qasm' )
qc_squander = QuantumCircuit.from_qasm_file( circuit_name +  '_squander.qasm' )
qc_qsearch  = QuantumCircuit.from_qasm_file( circuit_name +  '_qsearch.qasm' )


# generate random initial state on which we test the circuits
matrix_size = 1 << qc_original.num_qubits
initial_state_real = np.random.uniform(-1.0,1.0, (matrix_size,) )
initial_state_imag = np.random.uniform(-1.0,1.0, (matrix_size,) )
initial_state = initial_state_real + initial_state_imag*1j
initial_state = initial_state/np.linalg.norm(initial_state)


state_to_transform = initial_state.copy()
qc_original.initialize( state_to_transform )

state_to_transform = initial_state.copy()
qc_squander.initialize( state_to_transform )

state_to_transform = initial_state.copy()
qc_qsearch.initialize( state_to_transform )


# Execute and get the state vector
if qiskit_version[0] == '1':
	
	qc_original.save_statevector()
	qc_squander.save_statevector()
	qc_qsearch.save_statevector()

	backend = Aer.AerSimulator(method='statevector')

	# Execute and get the state vector
	result                     = backend.run(qc_original).result()
	transformed_state_original = np.array( result.get_statevector(qc_original) )

	result                     = backend.run(qc_squander).result()
	transformed_state_squander = np.array( result.get_statevector(qc_squander) )
	
	result                     = backend.run(qc_qsearch).result()
	transformed_state_qsearch  = np.array( result.get_statevector(qc_qsearch) )

	
       
        
elif qiskit_version[0] == '0':
	
	# Select the StatevectorSimulator from the Aer provider
	simulator = Aer.get_backend('statevector_simulator')	
		
	backend = Aer.get_backend('aer_simulator')

	# Execute and get the state vector
	result                     = execute(qc_original, simulator).result()
	transformed_state_original = np.array( result.get_statevector(qc_original) )

	result                     = execute(qc_squander, simulator).result()
	transformed_state_squander = np.array( result.get_statevector(qc_squander) )
	
	result                     = execute(qc_qsearch, simulator).result()
	transformed_state_qsearch  = np.array( result.get_statevector(qc_qsearch) )






overlap_squander = transformed_state_original.transpose().conjugate() @ transformed_state_squander
overlap_squander = overlap_squander * overlap_squander.conj()


overlap_qsearch = transformed_state_original.transpose().conjugate() @ transformed_state_qsearch
overlap_qsearch = overlap_qsearch * overlap_qsearch.conj()

print( 'The overlap of states obtained with the original and the squander compressed circuit: ',  overlap_squander )
print( 'The overlap of states obtained with the original and the qsearch compressed circuit: ',  overlap_qsearch )

