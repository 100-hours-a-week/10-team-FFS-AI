from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class BaseSchema(BaseModel):
    """모든 스키마의 기본 클래스
    
    Python에서는 snake_case를 사용하고, 
    JSON 데이터에서는 camelCase를 자동으로 지원하도록 설정합니다.
    """
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True
    )
