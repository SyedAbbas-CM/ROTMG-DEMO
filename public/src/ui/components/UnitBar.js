/**
 * Unit bar component for displaying and controlling units
 */
import { UIComponent } from '../UIManager.js';

export class UnitBar extends UIComponent {
  /**
   * Create a unit bar component
   * @param {Object} gameState - Game state reference
   * @param {Object} manager - UI manager reference
   */
  constructor(gameState, manager) {
    super(gameState, manager);
    
    this.selectedGroup = null;
    this.unitGroups = {};
    this.expanded = true;
  }
  
  /**
   * Initialize the component
   * @returns {HTMLElement} The component's DOM element
   */
  async init() {
    // Create panel container
    this.element = document.createElement('div');
    this.element.className = 'ui-unit-bar';
    this.element.style.position = 'absolute';
    this.element.style.bottom = '48px';
    this.element.style.right = '16px';
    this.element.style.width = '320px';
    this.element.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    this.element.style.border = '2px solid #444';
    this.element.style.color = 'white';
    this.element.style.zIndex = '10';
    this.element.style.pointerEvents = 'auto';
    
    // Create panel header
    const header = document.createElement('div');
    header.className = 'panel-header';
    header.style.display = 'flex';
    header.style.justifyContent = 'space-between';
    header.style.alignItems = 'center';
    header.style.padding = '2px 8px';
    header.style.borderBottom = '1px solid #333';
    header.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
    
    const title = document.createElement('div');
    title.className = 'panel-title selected-group-text';
    title.textContent = 'All Units';
    title.style.fontSize = '12px';
    title.style.color = '#aaa';
    
    const controls = document.createElement('div');
    controls.className = 'panel-controls';
    controls.style.display = 'flex';
    
    const expandBtn = document.createElement('button');
    expandBtn.className = 'btn-expand';
    expandBtn.innerHTML = '&minus;';
    expandBtn.title = 'Collapse';
    expandBtn.style.background = 'none';
    expandBtn.style.border = 'none';
    expandBtn.style.color = '#666';
    expandBtn.style.cursor = 'pointer';
    expandBtn.style.fontSize = '14px';
    expandBtn.style.width = '16px';
    expandBtn.style.height = '16px';
    expandBtn.style.display = 'flex';
    expandBtn.style.alignItems = 'center';
    expandBtn.style.justifyContent = 'center';
    expandBtn.style.marginLeft = '4px';
    
    const minimizeBtn = document.createElement('button');
    minimizeBtn.className = 'btn-minimize';
    minimizeBtn.innerHTML = '&#9776;'; // Menu icon
    minimizeBtn.title = 'Minimize';
    minimizeBtn.style.background = 'none';
    minimizeBtn.style.border = 'none';
    minimizeBtn.style.color = '#666';
    minimizeBtn.style.cursor = 'pointer';
    minimizeBtn.style.fontSize = '12px';
    minimizeBtn.style.width = '16px';
    minimizeBtn.style.height = '16px';
    minimizeBtn.style.display = 'flex';
    minimizeBtn.style.alignItems = 'center';
    minimizeBtn.style.justifyContent = 'center';
    minimizeBtn.style.marginLeft = '4px';
    
    const closeBtn = document.createElement('button');
    closeBtn.className = 'btn-close';
    closeBtn.innerHTML = '&times;';
    closeBtn.title = 'Close';
    closeBtn.style.background = 'none';
    closeBtn.style.border = 'none';
    closeBtn.style.color = '#666';
    closeBtn.style.cursor = 'pointer';
    closeBtn.style.fontSize = '12px';
    closeBtn.style.width = '16px';
    closeBtn.style.height = '16px';
    closeBtn.style.display = 'flex';
    closeBtn.style.alignItems = 'center';
    closeBtn.style.justifyContent = 'center';
    closeBtn.style.marginLeft = '4px';
    
    controls.appendChild(expandBtn);
    controls.appendChild(minimizeBtn);
    controls.appendChild(closeBtn);
    
    header.appendChild(title);
    header.appendChild(controls);
    
    // Quick selection buttons
    const quickSelect = document.createElement('div');
    quickSelect.className = 'quick-select';
    quickSelect.style.display = 'flex';
    quickSelect.style.padding = '4px';
    quickSelect.style.borderBottom = '1px solid #333';
    quickSelect.style.backgroundColor = 'rgba(0, 0, 0, 0.3)';
    
    const selectAll = document.createElement('button');
    selectAll.className = 'select-all';
    selectAll.textContent = 'Select All';
    selectAll.style.flex = '1';
    selectAll.style.marginRight = '4px';
    selectAll.style.padding = '2px 4px';
    selectAll.style.backgroundColor = '#222';
    selectAll.style.border = '1px solid #333';
    selectAll.style.color = '#ddd';
    selectAll.style.fontSize = '11px';
    selectAll.style.cursor = 'pointer';
    
    const selectNone = document.createElement('button');
    selectNone.className = 'select-none';
    selectNone.textContent = 'Select None';
    selectNone.style.flex = '1';
    selectNone.style.padding = '2px 4px';
    selectNone.style.backgroundColor = '#222';
    selectNone.style.border = '1px solid #333';
    selectNone.style.color = '#ddd';
    selectNone.style.fontSize = '11px';
    selectNone.style.cursor = 'pointer';
    
    quickSelect.appendChild(selectAll);
    quickSelect.appendChild(selectNone);
    
    // Groups list container
    const groupsContainer = document.createElement('div');
    groupsContainer.className = 'groups-container';
    groupsContainer.style.maxHeight = '120px';
    groupsContainer.style.overflowY = 'auto';
    groupsContainer.style.padding = '4px';
    groupsContainer.style.borderBottom = '1px solid #333';
    
    // Units container
    const unitsContainer = document.createElement('div');
    unitsContainer.className = 'units-container';
    unitsContainer.style.display = 'flex';
    unitsContainer.style.flexDirection = 'column';
    unitsContainer.style.maxHeight = '240px';
    unitsContainer.style.overflowY = 'auto';
    unitsContainer.style.padding = '4px';
    
    // Setup event handlers
    expandBtn.addEventListener('click', () => this.toggleExpand());
    minimizeBtn.addEventListener('click', () => this.minimize());
    closeBtn.addEventListener('click', () => this.hide());
    
    selectAll.addEventListener('click', () => this.selectAll());
    selectNone.addEventListener('click', () => this.selectNone());
    
    // Append elements to panel
    this.element.appendChild(header);
    this.element.appendChild(quickSelect);
    this.element.appendChild(groupsContainer);
    this.element.appendChild(unitsContainer);
    
    // Save references
    this.selectedGroupText = title;
    this.groupsContainer = groupsContainer;
    this.unitsContainer = unitsContainer;
    this.expandBtn = expandBtn;
    
    return this.element;
  }
  
  /**
   * Toggle expanded/collapsed state
   */
  toggleExpand() {
    this.expanded = !this.expanded;
    
    if (this.expanded) {
      this.unitsContainer.style.display = 'flex';
      this.expandBtn.innerHTML = '&minus;';
      this.expandBtn.title = 'Collapse';
    } else {
      this.unitsContainer.style.display = 'none';
      this.expandBtn.innerHTML = '+';
      this.expandBtn.title = 'Expand';
    }
  }
  
  /**
   * Minimize the panel
   */
  minimize() {
    this.hide();
    this.manager.trigger('minimizeComponent', { id: 'unitBar' });
  }
  
  /**
   * Select a group
   * @param {string|number} groupId - Group identifier
   */
  selectGroup(groupId) {
    this.selectedGroup = groupId;
    
    // Find the group
    const group = this.unitGroups[groupId];
    
    // Update title with group name
    if (group) {
      this.selectedGroupText.textContent = `Group: ${group.name} (${group.unitCount})`;
    } else if (groupId === 'all') {
      this.selectedGroupText.textContent = 'All Units';
    } else {
      this.selectedGroupText.textContent = 'Selected Units';
    }
    
    // Update UI to show selected group
    this.updateGroupSelection();
    
    // Update units display
    this.updateUnitsDisplay();
    
    // Trigger event for game to handle selection
    this.manager.trigger('selectGroup', { groupId });
  }
  
  /**
   * Update the visual selection of groups
   */
  updateGroupSelection() {
    const groupItems = this.groupsContainer.querySelectorAll('.group-item');
    
    groupItems.forEach(item => {
      const groupId = item.dataset.groupId;
      
      if (groupId === this.selectedGroup) {
        item.classList.add('selected');
        item.style.backgroundColor = '#333';
        item.style.borderColor = '#f59e0b';
      } else {
        item.classList.remove('selected');
        item.style.backgroundColor = '#222';
        item.style.borderColor = '#444';
      }
    });
  }
  
  /**
   * Select all units
   */
  selectAll() {
    this.selectGroup('all');
  }
  
  /**
   * Deselect all units except player
   */
  selectNone() {
    this.selectGroup('player');
  }
  
  /**
   * Update the units display based on selected group
   */
  updateUnitsDisplay() {
    this.unitsContainer.innerHTML = '';
    
    if (!this.selectedGroup || !this.unitGroups[this.selectedGroup]) {
      // If all units or invalid group, show empty state
      if (this.selectedGroup === 'all') {
        // Show all unit types
        this.renderAllUnitTypes();
      } else {
        const emptyState = document.createElement('div');
        emptyState.className = 'empty-units';
        emptyState.textContent = 'No units selected.';
        emptyState.style.color = '#666';
        emptyState.style.textAlign = 'center';
        emptyState.style.padding = '16px';
        this.unitsContainer.appendChild(emptyState);
      }
      return;
    }
    
    // Get selected group
    const group = this.unitGroups[this.selectedGroup];
    
    // Render units for this group
    this.renderGroupUnits(group);
  }
  
  /**
   * Render all unit types
   */
  renderAllUnitTypes() {
    // Get all unit types from all groups
    const unitTypes = {};
    
    // Collect units by type across all groups
    Object.values(this.unitGroups).forEach(group => {
      if (group.units) {
        Object.entries(group.units).forEach(([type, units]) => {
          if (!unitTypes[type]) {
            unitTypes[type] = [];
          }
          unitTypes[type] = unitTypes[type].concat(units);
        });
      }
    });
    
    // Render each unit type
    Object.entries(unitTypes).forEach(([type, units]) => {
      this.renderUnitTypeGroup(type, units);
    });
  }
  
  /**
   * Render units for a specific group
   * @param {Object} group - Group object
   */
  renderGroupUnits(group) {
    if (!group.units) return;
    
    // Render each unit type in the group
    Object.entries(group.units).forEach(([type, units]) => {
      this.renderUnitTypeGroup(type, units);
    });
  }
  
  /**
   * Render a group of units of the same type
   * @param {string} type - Unit type
   * @param {Array} units - Array of unit objects
   */
  renderUnitTypeGroup(type, units) {
    if (!units || units.length === 0) return;
    
    // Create type group container
    const typeGroup = document.createElement('div');
    typeGroup.className = 'unit-type-group';
    typeGroup.style.marginBottom = '8px';
    
    // Create type header
    const typeHeader = document.createElement('div');
    typeHeader.className = 'unit-type-header';
    typeHeader.textContent = type;
    typeHeader.style.fontSize = '12px';
    typeHeader.style.fontWeight = 'bold';
    typeHeader.style.color = '#aaa';
    typeHeader.style.marginBottom = '4px';
    typeHeader.style.padding = '2px 4px';
    typeHeader.style.borderBottom = '1px solid #333';
    
    // Create units wrapper
    const unitsWrapper = document.createElement('div');
    unitsWrapper.className = 'units-wrapper';
    unitsWrapper.style.display = 'flex';
    unitsWrapper.style.flexWrap = 'wrap';
    unitsWrapper.style.gap = '4px';
    
    // Add each unit
    units.forEach(unit => {
      const unitItem = document.createElement('div');
      unitItem.className = 'unit-item';
      unitItem.title = `${unit.name} (Level ${unit.level}) - ${type}`;
      unitItem.style.width = '74px';
      unitItem.style.height = '24px';
      unitItem.style.display = 'flex';
      unitItem.style.alignItems = 'center';
      unitItem.style.backgroundColor = '#222';
      unitItem.style.border = '1px solid #333';
      unitItem.style.padding = '2px';
      unitItem.style.cursor = 'pointer';
      
      // Unit icon
      const icon = document.createElement('div');
      icon.className = 'unit-icon';
      icon.style.width = '20px';
      icon.style.height = '20px';
      icon.style.display = 'flex';
      icon.style.alignItems = 'center';
      icon.style.justifyContent = 'center';
      icon.style.backgroundColor = '#333';
      
      // Set icon based on unit type
      let iconSymbol = '?';
      if (type === 'Infantry') {
        iconSymbol = '⛨'; // Shield
        icon.style.color = '#60a5fa'; // Blue
      } else if (type === 'Cavalry') {
        iconSymbol = '⚔'; // Swords
        icon.style.color = '#ef4444'; // Red
      } else if (type === 'Mages') {
        iconSymbol = '⚡'; // Lightning
        icon.style.color = '#f59e0b'; // Yellow
      } else if (type === 'Builders') {
        iconSymbol = '⚒'; // Hammer and pick
        icon.style.color = '#a3a3a3'; // Gray
      }
      
      icon.textContent = iconSymbol;
      
      // Health bar
      const health = document.createElement('div');
      health.className = 'unit-health';
      health.style.flex = '1';
      health.style.height = '4px';
      health.style.backgroundColor = '#333';
      health.style.marginLeft = '4px';
      health.style.marginRight = '4px';
      
      const healthBar = document.createElement('div');
      healthBar.className = 'unit-health-bar';
      healthBar.style.height = '100%';
      healthBar.style.backgroundColor = '#ef4444';
      healthBar.style.width = `${(unit.health / unit.maxHealth) * 100}%`;
      
      health.appendChild(healthBar);
      
      // Unit count
      const count = document.createElement('div');
      count.className = 'unit-count';
      count.textContent = unit.count || '1';
      count.style.fontSize = '10px';
      count.style.color = '#ddd';
      count.style.width = '16px';
      count.style.textAlign = 'center';
      
      unitItem.appendChild(icon);
      unitItem.appendChild(health);
      unitItem.appendChild(count);
      
      // Handle unit selection on click
      unitItem.addEventListener('click', () => {
        this.selectUnit(unit.id);
      });
      
      unitsWrapper.appendChild(unitItem);
    });
    
    typeGroup.appendChild(typeHeader);
    typeGroup.appendChild(unitsWrapper);
    
    this.unitsContainer.appendChild(typeGroup);
  }
  
  /**
   * Select a specific unit
   * @param {string|number} unitId - Unit identifier
   */
  selectUnit(unitId) {
    // Trigger event for game to handle unit selection
    this.manager.trigger('selectUnit', { unitId });
  }
  
  /**
   * Update the component with game state
   * @param {Object} gameState - Current game state
   */
  update(gameState) {
    // Update unit groups from game state
    if (gameState.unitGroups) {
      this.unitGroups = gameState.unitGroups;
      
      // If first time getting groups, update UI
      if (this.groupsContainer.children.length === 0) {
        this.renderGroups();
      }
      
      // Update selected group display
      this.updateUnitsDisplay();
    }
    
    // If selected group changed in game state
    if (gameState.selectedGroup !== undefined && 
        gameState.selectedGroup !== this.selectedGroup) {
      this.selectedGroup = gameState.selectedGroup;
      this.updateGroupSelection();
      this.updateUnitsDisplay();
    }
  }
  
  /**
   * Render the groups list
   */
  renderGroups() {
    this.groupsContainer.innerHTML = '';
    
    // Add "All" group
    const allGroup = document.createElement('div');
    allGroup.className = 'group-item';
    allGroup.dataset.groupId = 'all';
    allGroup.title = 'All Units';
    allGroup.style.display = 'flex';
    allGroup.style.alignItems = 'center';
    allGroup.style.padding = '4px';
    allGroup.style.marginBottom = '4px';
    allGroup.style.backgroundColor = '#222';
    allGroup.style.border = '1px solid #444';
    allGroup.style.cursor = 'pointer';
    
    const allIcon = document.createElement('div');
    allIcon.className = 'group-icon';
    allIcon.innerHTML = '★'; // Star
    allIcon.style.color = '#f59e0b';
    allIcon.style.marginRight = '8px';
    
    const allName = document.createElement('div');
    allName.textContent = 'All Units';
    allName.style.flex = '1';
    allName.style.fontSize = '12px';
    
    allGroup.appendChild(allIcon);
    allGroup.appendChild(allName);
    
    allGroup.addEventListener('click', () => this.selectGroup('all'));
    
    this.groupsContainer.appendChild(allGroup);
    
    // Add each group
    Object.entries(this.unitGroups).forEach(([id, group]) => {
      const groupItem = document.createElement('div');
      groupItem.className = 'group-item';
      groupItem.dataset.groupId = id;
      groupItem.title = `${group.name} (${group.unitCount} units)`;
      groupItem.style.display = 'flex';
      groupItem.style.alignItems = 'center';
      groupItem.style.padding = '4px';
      groupItem.style.marginBottom = '4px';
      groupItem.style.backgroundColor = '#222';
      groupItem.style.border = '1px solid #444';
      groupItem.style.cursor = 'pointer';
      
      // Group icon
      const icon = document.createElement('div');
      icon.className = 'group-icon';
      
      // Set icon based on primary class
      if (group.primaryClass === 'Knight') {
        icon.innerHTML = '⛨'; // Shield
        icon.style.color = '#60a5fa'; // Blue
      } else if (group.primaryClass === 'Archer') {
        icon.innerHTML = '◎'; // Target
        icon.style.color = '#22c55e'; // Green
      } else if (group.primaryClass === 'Priest') {
        icon.innerHTML = '♥'; // Heart
        icon.style.color = '#ef4444'; // Red
      } else if (group.primaryClass === 'Warrior') {
        icon.innerHTML = '⚔'; // Swords
        icon.style.color = '#f59e0b'; // Yellow
      } else {
        icon.innerHTML = '⚝'; // Star
        icon.style.color = '#a3a3a3'; // Gray
      }
      
      icon.style.marginRight = '8px';
      icon.style.fontSize = '14px';
      
      // Group name
      const name = document.createElement('div');
      name.className = 'group-name';
      name.textContent = group.name;
      name.style.flex = '1';
      name.style.fontSize = '12px';
      
      // Unit count
      const count = document.createElement('div');
      count.className = 'group-count';
      count.textContent = group.unitCount;
      count.style.marginLeft = '4px';
      count.style.fontSize = '10px';
      count.style.color = '#aaa';
      count.style.padding = '0px 4px';
      count.style.backgroundColor = '#333';
      count.style.borderRadius = '4px';
      
      groupItem.appendChild(icon);
      groupItem.appendChild(name);
      groupItem.appendChild(count);
      
      groupItem.addEventListener('click', () => this.selectGroup(id));
      
      this.groupsContainer.appendChild(groupItem);
    });
    
    // Select first group by default
    if (this.selectedGroup === null) {
      this.selectGroup('all');
    } else {
      this.updateGroupSelection();
    }
  }
} 