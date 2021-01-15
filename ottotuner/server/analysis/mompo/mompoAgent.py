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
from acme import adders as adderLib
from acme.adders.reverb import base

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

def task_reward_fn2(observation: tf.Tensor,
                   action: tf.Tensor,
                   reward: tf.Tensor) -> tf.Tensor:
  print(" in task_reward_fn_qphh ----------- observation and action are: ",str(observation), "\n", str(action))
  del observation, action
  return tf.stop_gradient(reward[:,1])

def task_reward_fn1(observation: tf.Tensor,
                   action: tf.Tensor,
                   reward: tf.Tensor) -> tf.Tensor:
  print(" in task_reward_fn_qphh ----------- observation and action are: ",str(observation), "\n", str(action))
  print("Shape of reward: ",str(reward.shape))
  del observation, action
  return tf.stop_gradient(reward[:,0])

def task_reward_fn(observation: tf.Tensor,
                   action: tf.Tensor,
                   reward: tf.Tensor) -> tf.Tensor:
  print(" in task_reward_fn_qphh ----------- observation and action are: ",str(observation), "\n", str(action))
  del observation, action
  return tf.stop_gradient(reward)


def make_objectives1(objectiveCount: int) -> Tuple[
    Sequence[mompoLearning.RewardObjective], Sequence[mompoLearning.QValueObjective]]:
  """Define the multiple objectives for the policy to learn."""
  task_reward = mompoLearning.RewardObjective(
      name='objective0',
      reward_fn=task_reward_fn1)
  action_norm = mompoLearning.QValueObjective(
      name='action_norm_q0',
      qvalue_fn=compute_action_norm)

  if objectiveCount > 1:
      task_reward1 = mompoLearning.RewardObjective(
              name='objective1',
              reward_fn=task_reward_fn2)
      action_norm1 = mompoLearning.QValueObjective(
              name='action_norm_q1',
              qvalue_fn=compute_action_norm)
      return [task_reward, task_reward1], [action_norm, action_norm1]
  else:
      print("In else, returning just 1")
      return [task_reward], [action_norm]



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


def mompoWrapper(mompoInput):
    recNum = mompoInput['y_matrix'].shape[0]
    dtype = np.float32
    mompoObjectives = mompoInput['objectives']
    print("The num of objectives is: ",str(mompoInput['objective_indexes'].shape))
    
    ### To make sure the wrapper is compatible for both single and multiple objectives
    ### Format the objectives received as accepted by the agent
    if mompoObjectives > 1:  ### Multi objective
        rewardArray = np.empty((recNum, mompoInput['objective_indexes'].shape[0]),dtype)
        for i in range(0,recNum):  ### Can have a nested loop here for more than 2 objectives
            rewardArray[i][0] = np.array([mompoInput['reward'][0][i]])
            rewardArray[i][1] = np.array([mompoInput['reward'][1][i]])
    else:       ### Single objective
        print("The shape is: ",str(mompoInput['objective_indexes'].shape))
        rewardArray = np.array([[el] for el in mompoInput['reward']] )
    print("The reward array is: ",str(rewardArray.shape))       ### Final reward array as accepted by the agent

    ### Creating objectives
    reward_objectives, qvalue_objectives = make_objectives1(objectiveCount=mompoObjectives)
    #reward_objectives, qvalue_objectives = make_objectives()
    num_critic_heads = len(reward_objectives)

    ### Assign, observations, rewards, discounts and actions to the variable as below and create an env object. 
    observations = specs.Array(mompoInput['y_matrix'][0].shape, dtype)  ### metric is the observation
    rewards = specs.Array(mompoInput['objective_indexes'].shape, dtype)
    #rewards = specs.Array((), dtype)
    discounts = specs.BoundedArray((), dtype, 0.0, 1.0)
    actions = specs.Array(mompoInput['X_matrix'][0].shape, dtype)   #### Conf knobs are actions

    ### Check and assign values here
    print("The shape of the rewards: ",str(rewards))
    environment = fakes.Environment(spec= specs.EnvironmentSpec(
            observations=observations,
            actions=actions,
            rewards=rewards,
            discounts=discounts), episode_length=10)
    #### Or, just create the spec object without env using the variables and statements above in the comments.
    ### But, environment might be needed for the loop
    spec = specs.make_environment_spec(environment)
    timestep = environment.reset()

    print('actions:\n', spec.actions, '\n')
    print('observations:\n', spec.observations, '\n')
    print('rewards:\n', spec.rewards, '\n')
    print('discounts:\n', spec.discounts, '\n')

    ### Create dataset
    # Create a replay server to add data to.
    replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=1000000,
        rate_limiter=reverb.rate_limiters.MinSize(min_size_to_sample=1),
        signature=adders.NStepTransitionAdder.signature(spec))
    server = reverb.Server([replay_table], port=None)

    ### Create a client to add data into the replay server
    address = f'localhost:{server.port}'
    replayTabClient = reverb.Client(address)
    myadder = adders.NStepTransitionAdder(
        client=replayTabClient,
        n_step=5,
        discount=0.99)

    recNum = mompoInput['y_matrix'].shape[0]
    print("Num of recs: ",str(recNum))
    ### ddpg.add_sample(prev_normalized_metric_data, knob_data, reward, normalized_metric_data)
    ### Adding data into the replay server
    with replayTabClient.writer(max_sequence_length=mompoInput['y_matrix'].shape[0]) as writer: 
        for i in range (1,recNum):
            rewardVal = rewardArray[i]
            #rewardVal = 2
            dataRow = (np.float32(mompoInput['y_matrix'][i-1]),     ### prev observation
                    np.float32(mompoInput['X_matrix'][i]),          ### Current action
                    np.float32(rewardVal),                          ### Current reward
                    np.float32(4),                                  ### Current discount
                    np.float32(mompoInput['y_matrix'][i]))          ### Current observation
            #replayTabClient.insert(dataRow, priorities={adders.DEFAULT_PRIORITY_TABLE: 1.0})
            writer.append(dataRow)
        writer.create_item(adders.DEFAULT_PRIORITY_TABLE, num_timesteps=recNum-1, priority=1.0)

    # The dataset object to learn from.
    dataset = datasets.make_reverb_dataset(
        table=adders.DEFAULT_PRIORITY_TABLE,
        server_address=address,
        batch_size=512,
        prefetch_size=4)

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
        dataset = dataset,
        adder = myadder,
        batch_size=10,
        samples_per_insert=2,
        min_replay_size=10)

    ### Run the environment and get the next action (next configuration)
    loop = acme.EnvironmentLoop(environment, agent)  #### Here, Agent is gng as an actor
    res = loop.run(num_episodes=4, num_steps=None)  ### Either num_episodes or num_steps should be None
    finalAction = agent.select_action(timestep.observation)
    print("--------------------------- After loop------ Ending with res: ",str(finalAction))
    return finalAction

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
            #print("The shape of the array now is: ", str(ipContents[key].shape))
            print("the keys and value types are: ",str(key),", ",str(ipContents[key].shape))

    #print ("len of the array: 1D ", str(len(ipContents['X_matrix'])) )
    #print ("len of the array: 2D ", str(len(ipContents['X_matrix'][0])) )

    print("\n\nFinished reading\n")

    retVal = np.array(mompoWrapper(ipContents)).tolist()
    
    mompoOutput = {}
    mompoOutput['test'] = "lakssasljdsajdasjasjdsdjaskjdhsjhdaskhdaskdhsa"
    mompoOutput['nextConfig'] = retVal

    file1 = open("/home/mrd/Desktop/OttoTuner/otterTuneCode/ottertune/server/analysis/mompo/output.txt","w")
    json.dump(mompoOutput, file1)
    file1.close();

    print("\n\nFinished Writing\n")

    return

main();
