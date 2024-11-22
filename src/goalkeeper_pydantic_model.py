from enum import Enum
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator, computed_field


# Enumerations (remain the same as in previous version)
class GoalStatus(str, Enum):
    NOT_STARTED = "NotStarted"
    IN_PROGRESS = "InProgress"
    COMPLETED = "Completed"
    BLOCKED = "Blocked"

class ValueDomain(str, Enum):
    PERSONAL_GROWTH = "Personal Growth"
    PROFESSIONAL_DEVELOPMENT = "Professional Development"
    RELATIONSHIPS = "Relationships"
    HEALTH_WELLNESS = "Health & Wellness"
    COMMUNITY_IMPACT = "Community Impact"
    SPIRITUALITY = "Spirituality"
    FINANCIAL_INDEPENDENCE = "Financial Independence"

class MindsetType(str, Enum):
    GROWTH = "GrowthMindset"
    FIXED = "FixedMindset"

# Core Models with Pydantic v2 Features
class CoreValue(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=False,
        validate_assignment=True,
        extra='ignore'
    )
    
    name: str
    description: Optional[str] = None
    domain: ValueDomain
    importance_score: float = Field(ge=0, le=10, default=5.0)

class PerformanceMetric(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    name: str
    current_value: float
    target_value: float
    measurement_unit: str
    
    @computed_field
    @property
    def progress_percentage(self) -> float:
        return (self.current_value / self.target_value) * 100 if self.target_value != 0 else 0.0

class ValueBasedGoal(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        extra='ignore'
    )
    
    title: str
    description: Optional[str] = None
    start_date: date
    target_completion_date: date
    status: GoalStatus = GoalStatus.NOT_STARTED
    core_values: List[CoreValue] = []
    performance_metrics: List[PerformanceMetric] = []
    
    @field_validator('target_completion_date')
    @classmethod
    def validate_completion_date(cls, v, info):
        start_date = info.data.get('start_date')
        if start_date and v < start_date:
            raise ValueError("Target completion date must be after start date")
        return v
    
    @computed_field
    @property
    def value_alignment_score(self) -> float:
        if not self.core_values:
            return 0.0
        return sum(value.importance_score for value in self.core_values) / (len(self.core_values) * 10) * 100

class Mindset(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    type: MindsetType
    key_characteristics: List[str] = []
    development_areas: List[str] = []

class Intervention(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    title: str
    description: Optional[str] = None
    start_date: date
    duration_days: int
    associated_goals: List[str] = []
    
    @computed_field
    @property
    def is_active(self) -> bool:
        current_date = date.today()
        end_date = date(
            self.start_date.year, 
            self.start_date.month, 
            self.start_date.day
        )
        return (current_date >= self.start_date) and (current_date <= end_date)

class DomainMindset(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    type: MindsetType
    confidence_level: float = 0.0  # 0-1 scale representing certainty
    key_characteristics: List[str] = []
    growth_potential: float = 0.0  # 0-1 scale of potential for mindset development

class Individual(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    name: str
    domain_mindsets: Dict[ValueDomain, DomainMindset] = {}
    
    def update_domain_mindset(
        self, 
        domain: ValueDomain, 
        mindset_type: MindsetType, 
        confidence: float = 0.5,
        characteristics: Optional[List[str]] = None,
        growth_potential: float = 0.5
    ):
        """
        Update or create a domain-specific mindset
        
        Args:
            domain: The specific value domain
            mindset_type: Growth or Fixed mindset
            confidence: Confidence in the mindset assessment (0-1)
            characteristics: Specific mindset characteristics
            growth_potential: Potential for mindset development (0-1)
        """
        self.domain_mindsets[domain] = DomainMindset(
            type=mindset_type,
            confidence_level=confidence,
            key_characteristics=characteristics or [],
            growth_potential=growth_potential
        )
    
    def get_domain_mindset(self, domain: ValueDomain) -> Optional[DomainMindset]:
        """
        Retrieve mindset for a specific domain
        """
        return self.domain_mindsets.get(domain)
    
    def assess_overall_mindset(self) -> MindsetType:
        """
        Determine overall mindset based on domain-specific mindsets
        """
        if not self.domain_mindsets:
            return MindsetType.FIXED
        
        # Count growth mindset domains
        growth_domains = sum(
            1 for mindset in self.domain_mindsets.values() 
            if mindset.type == MindsetType.GROWTH
        )
        
        # If majority of domains show growth mindset, return Growth
        return (MindsetType.GROWTH if growth_domains > len(self.domain_mindsets) / 2 
                else MindsetType.FIXED)

#Goal Planning and Obstacle Management

class ObstacleType(str, Enum):
    INTERNAL = "Internal"
    EXTERNAL = "External"
    RESOURCE_BASED = "Resource-Based"
    SKILL_BASED = "Skill-Based"

class ObstacleSeverity(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class Obstacle(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    description: str
    type: ObstacleType
    severity: ObstacleSeverity
    impact_on_goal: float = Field(ge=0, le=1)  # 0-1 scale of goal impediment

class Solution(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    description: str
    estimated_effectiveness: float = Field(ge=0, le=1)
    required_resources: List[str] = []
    estimated_time_investment: Optional[timedelta] = None

class ActionStep(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    description: str
    target_start_date: date
    target_completion_date: date
    status: str = "Pending"
    dependencies: List[str] = []
    allocated_resources: List[str] = []

class GoalPlan(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    goal_title: str
    action_steps: List[ActionStep] = []
    obstacles: List[Obstacle] = []
    solutions: Dict[str, Solution] = {}
    
    def add_obstacle(self, obstacle: Obstacle):
        self.obstacles.append(obstacle)
    
    def propose_solution(self, obstacle: Obstacle) -> Solution:
        """
        Generate potential solution for an obstacle
        """
        solution_strategies = {
            ObstacleType.INTERNAL: self._internal_obstacle_solution,
            ObstacleType.EXTERNAL: self._external_obstacle_solution,
            ObstacleType.RESOURCE_BASED: self._resource_obstacle_solution,
            ObstacleType.SKILL_BASED: self._skill_obstacle_solution
        }
        
        strategy = solution_strategies.get(obstacle.type, self._generic_solution)
        return strategy(obstacle)
    
    def _internal_obstacle_solution(self, obstacle: Obstacle) -> Solution:
        return Solution(
            description=f"Mindset and personal development approach to {obstacle.description}",
            estimated_effectiveness=0.7,
            required_resources=["Personal coaching", "Self-reflection time"],
            estimated_time_investment=timedelta(days=30)
        )
    
    def _external_obstacle_solution(self, obstacle: Obstacle) -> Solution:
        return Solution(
            description=f"Strategic approach to external challenge: {obstacle.description}",
            estimated_effectiveness=0.6,
            required_resources=["Network consultation", "External expert advice"],
            estimated_time_investment=timedelta(days=45)
        )
    
    def _resource_obstacle_solution(self, obstacle: Obstacle) -> Solution:
        return Solution(
            description=f"Resource acquisition strategy for {obstacle.description}",
            estimated_effectiveness=0.8,
            required_resources=["Funding", "Equipment", "Additional personnel"],
            estimated_time_investment=timedelta(days=60)
        )
    
    def _skill_obstacle_solution(self, obstacle: Obstacle) -> Solution:
        return Solution(
            description=f"Skill development plan for {obstacle.description}",
            estimated_effectiveness=0.9,
            required_resources=["Training", "Online courses", "Mentorship"],
            estimated_time_investment=timedelta(days=90)
        )
    
    def _generic_solution(self, obstacle: Obstacle) -> Solution:
        return Solution(
            description=f"Generic approach to {obstacle.description}",
            estimated_effectiveness=0.5,
            required_resources=["Adaptive problem-solving"],
            estimated_time_investment=timedelta(days=30)
        )
# Particpant Model
class ParticipantType(str, Enum):
    USER = "Human"
    ASSISTANT = "AI"
    REFERENCED = "Referenced"

class Document(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    content: str
    timestamp: datetime
    source: ParticipantType
    message_id: str = Field(description="Unique identifier for the message")
    conversation_id: str = Field(description="Identifier for the conversation thread")
    referenced_entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Dictionary mapping entity types to lists of entity identifiers referenced in the message"
    )
    
    @field_validator('source')
    @classmethod
    def validate_source(cls, v):
        if v not in [ParticipantType.USER, ParticipantType.ASSISTANT]:
            raise ValueError("Document source must be either Human or AI")
        return v

class Participant(BaseModel):
    """Base class for all participants in the system"""
    model_config = ConfigDict(validate_assignment=True)
    
    name: str
    participant_id: str = Field(description="Unique identifier for the participant")
    participant_type: ParticipantType
    domain_mindsets: Dict[ValueDomain, DomainMindset] = {}
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: Optional[datetime] = None
    
    @computed_field
    @property
    def is_active_participant(self) -> bool:
        return self.participant_type in [ParticipantType.USER, ParticipantType.ASSISTANT]

class ChatParticipant(Participant):
    """Specific model for User and AI participants who can author documents"""
    model_config = ConfigDict(validate_assignment=True)
    
    authored_documents: List[str] = Field(
        default_factory=list,
        description="List of document IDs authored by this participant"
    )
    conversations: List[str] = Field(
        default_factory=list,
        description="List of conversation IDs this participant is involved in"
    )
    
    @field_validator('participant_type')
    @classmethod
    def validate_participant_type(cls, v):
        if v == ParticipantType.REFERENCED:
            raise ValueError("ChatParticipant cannot be of type Referenced")
        return v

class ReferencedIndividual(Participant):
    """Model for individuals referenced in conversations"""
    model_config = ConfigDict(validate_assignment=True)
    
    mentioned_in_documents: List[str] = Field(
        default_factory=list,
        description="List of document IDs where this individual is referenced"
    )
    
    @field_validator('participant_type')
    @classmethod
    def validate_participant_type(cls, v):
        if v != ParticipantType.REFERENCED:
            raise ValueError("ReferencedIndividual must be of type Referenced")
        return v

class OtherEntityType(str, Enum):
    ORGANIZATION = "Organization"
    LOCATION = "Location"
    INSTITUTION = "Institution"
    CONCEPT = "Concept"
    SYSTEM = "System"
    OTHER = "Other"

class OtherEntity(BaseModel):
    """
    Represents entities that do not fit into primary participant or specific node types
    """
    model_config = ConfigDict(validate_assignment=True)
    
    name: str
    entity_type: OtherEntityType
    sub_type: Optional[str] = None
    description: Optional[str] = None
    first_mentioned_in: Optional[str] = Field(
        default=None, 
        description="Document ID where entity was first referenced"
    )
    referenced_documents: List[str] = Field(
        default_factory=list, 
        description="List of document IDs mentioning this entity"
    )
    
    def add_reference(self, document_id: str):
        """
        Track documents referencing this entity
        """
        if document_id not in self.referenced_documents:
            self.referenced_documents.append(document_id)
        
        if not self.first_mentioned_in:
            self.first_mentioned_in = document_id
            
# Example Usage
# Example Usage
def create_performance_coaching_example():
    # Create core values
    personal_growth_value = CoreValue(
        name="Continuous Learning", 
        domain=ValueDomain.PERSONAL_GROWTH,
        importance_score=9.0
    )
    
    # Create a goal
    leadership_goal = ValueBasedGoal(
        title="Develop Authentic Leadership",
        start_date=date.today(),
        target_completion_date=date(2025, 12, 31),
        core_values=[personal_growth_value],
        performance_metrics=[
            PerformanceMetric(
                name="Leadership Effectiveness",
                current_value=6.5,
                target_value=9.0,
                measurement_unit="Scale"
            )
        ]
    )
    
    # Create an individual
    coaching_participant = Individual(
        name="Alex Rodriguez",
        email="alex.rodriguez@example.com",
        core_values=[personal_growth_value],
        goals=[leadership_goal],
        mindset=Mindset(
            type=MindsetType.GROWTH,
            key_characteristics=["Embraces challenges", "Persistent"]
        )
    )
    
    return coaching_participant

# Demonstration
if __name__ == "__main__":
    participant = create_performance_coaching_example()
    print(f"Participant: {participant.name}")
    print(f"Goal Completion Rate: {participant.goal_completion_rate}%")


def create_domain_specific_mindset_example():
    entrepreneur = Individual(name="Innovative Entrepreneur")
    
    # Set domain-specific mindsets
    entrepreneur.update_domain_mindset(
        domain=ValueDomain.PROFESSIONAL_DEVELOPMENT,
        mindset_type=MindsetType.GROWTH,
        confidence=0.8,
        characteristics=[
            "Seeks continuous learning", 
            "Embraces professional challenges"
        ],
        growth_potential=0.9
    )
    
    entrepreneur.update_domain_mindset(
        domain=ValueDomain.FINANCIAL_INDEPENDENCE,
        mindset_type=MindsetType.FIXED,
        confidence=0.6,
        characteristics=[
            "Risk-averse", 
            "Prefers stable income strategies"
        ],
        growth_potential=0.4
    )
    
    # Assess overall and domain-specific mindsets
    print(f"Overall Mindset: {entrepreneur.assess_overall_mindset()}")
    
    # Retrieve specific domain mindset
    prof_dev_mindset = entrepreneur.get_domain_mindset(ValueDomain.PROFESSIONAL_DEVELOPMENT)
    print(f"Professional Development Mindset: {prof_dev_mindset.type}")
    print(f"Growth Potential: {prof_dev_mindset.growth_potential}")
    
    return entrepreneur

# Run example
example_entrepreneur = create_domain_specific_mindset_example()