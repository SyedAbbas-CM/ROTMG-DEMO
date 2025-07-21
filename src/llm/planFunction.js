// File: src/llm/planFunction.js

export const issueActionsFn = {
  name: 'issue_actions',
  description:
    'Return EITHER a quick { actions:[…] } plan OR a full behaviour `script` object.',
  parameters: {
    type: 'object',
    properties: {
      intent: {
        type: 'string',
        description: 'Optional human-readable summary of intent',
      },
      priority: {
        type: 'string',
        enum: ['low', 'medium', 'high'],
        description: 'Urgency hint (high = clear existing queue first)',
      },
      actions: {
        type: 'array',
        description: 'One-shot list of atomic ability calls',
        items: {
          type: 'object',
          properties: {
            ability: {
              type: 'string',
              enum: [
                'dash',
                'radial_burst',
                'cone_aoe',
                'wait',
                'spawn_minions',
                'reposition',
                'taunt',
                'spawn_formation',
                'teleport_to_player',
                'dynamic_movement',
                'charge_attack',
                'pattern_shoot',
                'summon_orbitals',
                'heal_self',
                'shield_phase',
                'effect_aura',
                'conditional_trigger',
                'environment_control',
                'projectile_spread'
              ]
            },
            args:    { type: 'object' },
          },
          required: ['ability'],
        },
      },
      script: {
        type: 'object',
        description: 'Full behaviour DSL v1 object (entry, states, limits …)',
      },
    },
    // You can enforce one of these client-side after parsing the result:
    // required: ['actions'] 
  },
};
