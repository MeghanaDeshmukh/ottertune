from typing import Dict, Sequence, Tuple

from absl.testing import absltest
from absl.testing import parameterized
import acme
from acme import specs
from acme.testing import fakes
from acme.tf import networks
import numpy as np
import sonnet as snt
import tensorflow as tf
import agent as mompo
import learning as mompoLearning
import sys
import ast
import json
import reverb
from acme.adders import reverb as adders
from acme import datasets

from absl.testing import absltest

import acme
from acme import specs
from acme.agents.tf import dqn
from acme.testing import fakes

import numpy as np
import sonnet as snt



def make_networks(
    action_spec: specs.Array,
    num_critic_heads: int,
    policy_layer_sizes: Sequence[int] = (300, 200),
    critic_layer_sizes: Sequence[int] = (400, 300),
    num_layers_shared: int = 1,
    distributional_critic: bool = True,
    vmin: float = -150.,
    vmax: float = 150.,
    num_atoms: int = 51,
) -> Dict[str, snt.Module]:
  """Creates networks used by the agent."""

  num_dimensions = np.prod(action_spec.shape, dtype=int)

  policy_network = snt.Sequential([
      networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
      networks.MultivariateNormalDiagHead(
          num_dimensions,
          tanh_mean=False,
          init_scale=0.69)
  ])

  if not distributional_critic:
    critic_layer_sizes = list(critic_layer_sizes) + [1]

  if not num_layers_shared:
    # No layers are shared
    critic_network_base = None
  else:
    critic_network_base = networks.LayerNormMLP(
        critic_layer_sizes[:num_layers_shared], activate_final=True)
  critic_network_heads = [
      snt.nets.MLP(critic_layer_sizes, activation=tf.nn.elu,
                   activate_final=False)
      for _ in range(num_critic_heads)]
  if distributional_critic:
    critic_network_heads = [
        snt.Sequential([
            c, networks.DiscreteValuedHead(vmin, vmax, num_atoms)
        ]) for c in critic_network_heads]
  # The multiplexer concatenates the (maybe transformed) observations/actions.
  critic_network = snt.Sequential([
      networks.CriticMultiplexer(
          critic_network=critic_network_base),
      networks.Multihead(network_heads=critic_network_heads),
  ])
  return {
      'policy': policy_network,
      'critic': critic_network,
  }

def compute_action_norm(target_pi_samples: tf.Tensor,
                        target_q_target_pi_samples: tf.Tensor) -> tf.Tensor:
  """Compute Q-values for the action norm objective from action samples."""
  del target_q_target_pi_samples
  action_norm = tf.norm(target_pi_samples, ord=2, axis=-1)
  return tf.stop_gradient(-1 * action_norm)



#### MRD Change this to return QphH and tpmC
#### Input sends observation, action and final reward as input
#### From the observation, get tpmc or qphh. 
#### Subtract that val from the final reward to get the other value.
#### How to send two reward objectives?

def task_reward_fn(observation: tf.Tensor,
                   action: tf.Tensor,
                   reward: tf.Tensor) -> tf.Tensor:
  print(" in task_reward_fn_qphh ----------- observation and action are: ",str(observation), "\n", str(action))
  del observation, action
  return tf.stop_gradient(reward)


def make_objectives() -> Tuple[
    Sequence[mompoLearning.RewardObjective], Sequence[mompoLearning.QValueObjective]]:
  """Define the multiple objectives for the policy to learn."""
  task_reward = mompoLearning.RewardObjective(
      name='task',
      reward_fn=task_reward_fn)
  action_norm = mompoLearning.QValueObjective(
      name='action_norm_q',
      qvalue_fn=compute_action_norm)
  return [task_reward], [action_norm]

def _make_network(action_spec: specs.DiscreteArray) -> snt.Module:
  return snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([50, 50, action_spec.num_values]),
  ])

def dqnWrapper(mompoInput):
    # Create a fake environment to test with.   ### This needs to be updated.
    #environment = fakes.ContinuousEnvironment(episode_length=10)
    ### MRD --- Change the above line as below.
    

    ### Assign, observations, rewards, discounts and actions to the variable as below and create an env object. 
    dtype = np.float32
    observations = specs.Array(mompoInput['X_matrix'][0].shape, dtype) 
    rewards = specs.Array((), dtype)
    discounts = specs.BoundedArray((), dtype, 0.0, 1.0)
    #actions = specs.BoundedArray(action_shape, dtype, -1.0, 1.0)
    actions = specs.Array(mompoInput['y_matrix'][0].shape, dtype)

    ### Check and assign values here
    print("Before assigning observations and actions are: ",str(observations), "\n", str(actions))
#    observations = mompoInput['X_matrix'];
#    actions = mompoInput['y_matrix']
    environment = fakes.DiscreteEnvironment(
        num_actions=5,
        num_observations=10,
        obs_dtype=np.float32,
        episode_length=10)
    spec = specs.make_environment_spec(environment)

    # Construct the agent.
    agent = dqn.DQN(
        environment_spec=spec,
        network=_make_network(spec.actions),
        batch_size=10,
        samples_per_insert=2,
        min_replay_size=10)

    loop = acme.EnvironmentLoop(environment, agent)  #### Here, Agent is gng as an actor
    res = loop.run(num_episodes=None, num_steps=4)
#    res = loop.should_terminate(episode_count =1 , step_count=1)
    print("--------------------------- After loop------ Ending with res: ",str(res))
    return
    print("--------------------------- Before Timestep: ", str(environment._spec.observations))
    print('actions:\n', spec.actions, '\n')
    print('observations:\n', spec.observations, '\n')
    print('rewards:\n', spec.rewards, '\n')
    print('discounts:\n', spec.discounts, '\n')

    ### Create dataset
    # Create a replay server to add data to.
    replay_table = reverb.Table(
        name='ottoTuner',
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=1000000,
        rate_limiter=reverb.rate_limiters.MinSize(min_size_to_sample=1),
        signature=adders.NStepTransitionAdder.signature(spec))
    server = reverb.Server([replay_table], port=None)

    # The adder is used to insert observations into replay.
    ### MRD --- This needs to be changed as this is what is adding data into the replay table.
    ### Instead add the ottotuner data here.
    address = f'localhost:{server.port}'
    '''adder = adders.NStepTransitionAdder(
        client=reverb.Client(address),
        n_step=5,
        discount=0.99)
    '''
    # The dataset object to learn from.
    dataset = datasets.make_reverb_dataset(
        table='ottoTuner',
        server_address=address,
        batch_size=512,
        prefetch_size=4)

    # Create objectives.   ### This needs to be changed.
    reward_objectives, qvalue_objectives = make_objectives()
    num_critic_heads = len(reward_objectives)

    # Create networks.  ### This is taken care of once the env is ready
    agent_networks = make_networks(
        spec.actions, num_critic_heads=num_critic_heads,
        distributional_critic=True)

    # Construct the agent.
    agent = mompo.MultiObjectiveMPO(
        reward_objectives,
        qvalue_objectives,
        spec,
        policy_network=agent_networks['policy'],
        critic_network=agent_networks['critic'],
   #     dataset = dataset,
        batch_size=10,
        samples_per_insert=2,
        min_replay_size=10)

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent)  #### Here, Agent is gng as an actor
    res = loop.run(num_episodes=None, num_steps=4)
#    res = loop.should_terminate(episode_count =1 , step_count=1)
    print("--------------------------- After loop------ Ending with res: ",str(res))
    return
    print("--------------------------- Before Timestep: ", str(environment._spec.observations))
    ## Trying to implement here what happens in the loop
    timestep = environment.reset()
    print("---------------------------- time step obtained and it is: ",str(type(timestep)), str(timestep))
#    timestep.observation = mompoInput['X_matrix']
    action = agent.select_action(timestep.observation)
    print(" ---------------------- action obtained and it is: ",str(type(action)), str(action.shape), str(timestep.observation.shape))
    timestep = environment.step(action) ### Should change the step function as well.

    agent.observe(action, next_timestep=timestep)
#    if self._should_update:
#        self._actor.update()
    
    ### In order to get the next step, 
    ### The below commands might work
    '''
    timestep = self._environment.reset()
    # Make the first observation.
    self._actor.observe_first(timestep)
    timestep = self._environment.step(action)
    agent.select_action(timestep.observation) 
    '''

    return


def main():
    print("In mompoAgent.py -------- MRD")
    ipFileName = "/home/mrd/Desktop/OttoTuner/otterTuneCode/ottertune/server/analysis/mompo/input.txt"

    f = open(ipFileName, "r")
    ipContents = f.read()
    f.close()

    with open(ipFileName) as f:
      ipContents = json.load(f)

    #print("input is: ",str(ipContents))
    print("input is: ",str(type(ipContents)))
    #ipDict = ast.literal_eval(ipContents)
    #ipDict = eval(ipContents)

    #print("the keys are: ",str(ipDict.keys()))
    #print("the keys are: ",str(ipContents.keys()))
    for key in ipContents.keys():
        print("the keys and value types are: ",str(key),", ",str(type(ipContents[key])))
        if isinstance(ipContents[key], list):
     #       print("This is a list. Converting it into a nd array now.")
            ipContents[key] = np.array(ipContents[key])
            print("The shape of the array now is: ", str(ipContents[key].shape))

    #print ("len of the array: 1D ", str(len(ipContents['X_matrix'])) )
    #print ("len of the array: 2D ", str(len(ipContents['X_matrix'][0])) )

    print("\n\nFinished reading\n")

    retVal = dqnWrapper(ipContents)
    
    mompoOutput = {}
    mompoOutput['test'] = "lakssasljdsajdasjasjdsdjaskjdhsjhdaskhdaskdhsa"

    file1 = open("/home/mrd/Desktop/OttoTuner/otterTuneCode/ottertune/server/analysis/mompo/output.txt","w")
    json.dump(mompoOutput, file1)
    file1.close();

    print("\n\nFinished Writing\n")

    return

main();
