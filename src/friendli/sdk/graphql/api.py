"""Code generated from api.graphql via strawberry_codegen."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import NewType

from pydantic import BaseModel, Field

from ..stub import AbstractAsyncStub, AbstractSyncStub, TypeName

UserContextGql = """
query UserContext($conn: BidirectionalConnectionInput, $sorts: ClientUserTeamSortsInput) {
  clientUser {
    teams(conn: $conn, sorts: $sorts) {
      totalCount
      edges {
        node {
          id
          name
        }
        productAccess {
          container
          dedicatedEndpoints
          serverlessEndpoints
        }
        role
        joinedAt
        default
      }
    }
  }
}
"""


AdapterPushStartGql = """
mutation AdapterPushStart($input: DedicatedModelPushAdapterStartInput!) {
  dedicatedModelPushAdapterStart(input: $input) {
    __typename
    ... on DedicatedModelPushAdapterStartSuccess {
      adapter {
        name
        id
        createdAt
      }
      uploadPlan {
        adapterConfig {
          required
        }
        tokenizerConfig {
          required
        }
        safetensors {
          required
        }
      }
    }
    ... on UserPermissionError {
      message
    }
  }
}
"""


AdapterPushCompleteGql = """
mutation AdapterPushComplete($input: DedicatedModelPushAdapterCompleteInput!) {
  dedicatedModelPushAdapterComplete(input: $input) {
    __typename
    ... on UserPermissionError {
      message
    }
    ... on DedicatedModelPushAdapterCompleteSuccess {
      adapter {
        name
        id
        createdAt
      }
    }
  }
}
"""


FilePushStartGql = """
mutation FilePushStart($input: DedicatedModelPushFileStartInput!) {
  dedicatedModelPushFileStart(input: $input) {
    __typename
    ... on UserPermissionError {
      message
    }
    ... on DedicatedModelPushFileStartSuccess {
      uploadInfo {
        uploadUrl
        uploadBody
      }
    }
    ... on DedicatedModelPushFileStartAlreadyExistError {
      message
    }
  }
}
"""


FilePushCompleteGql = """
mutation FilePushComplete($input: DedicatedModelPushFileCompleteInput!) {
  dedicatedModelPushFileComplete(input: $input) {
    __typename
    ... on DedicatedModelPushFileCompleteSuccess {
      ok
    }
    ... on UserPermissionError {
      message
    }
  }
}
"""


BasePushStartGql = """
mutation BasePushStart($input: DedicatedModelPushBaseStartInput!) {
  dedicatedModelPushBaseStart(input: $input) {
    ... on DedicatedModelPushBaseStartSuccess {
      model {
        name
        id
        digest
        createdAt
      }
      uploadPlan {
        config {
          required
        }
        tokenizer {
          required
        }
        tokenizerConfig {
          required
        }
        specialTokensMap {
          required
        }
        safetensors {
          required
        }
      }
    }
  }
}
"""


BasePushCompleteGql = """
mutation BasePushComplete($input: DedicatedModelPushBaseCompleteInput!) {
  dedicatedModelPushBaseComplete(input: $input) {
    ... on DedicatedModelPushBaseCompleteSuccess {
      model {
        updatedAt
        name
        id
        digest
        createdAt
      }
    }
  }
}
"""


ChunkGroupCreateGql = """
mutation ChunkGroupCreate($input: DedicatedModelCreateChunkGroupInput!) {
  dedicatedModelCreateChunkGroup(input: $input) {
    ... on DedicatedModelCreateChunkGroupSuccess {
      chunkGroupId
    }
  }
}
"""


ChunkGroupCommitGql = """
mutation ChunkGroupCommit($input: DedicatedModelCommitChunkGroupInput!) {
  dedicatedModelCommitChunkGroup(input: $input) {
    ... on DedicatedModelCommitChunkGroupSuccess {
      ok
    }
  }
}
"""


ChunkPushStartGql = """
mutation ChunkPushStart($input: DedicatedModelPushChunkStartInput!) {
  dedicatedModelPushChunkStart(input: $input) {
    __typename
    ... on DedicatedModelPushChunkStartSuccess {
      uploadUrl
    }
    ... on DedicatedModelPushChunkStartAlreadyExistError {
      message
    }
  }
}
"""


ChunkPushCompleteGql = """
mutation ChunkPushComplete($input: DedicatedModelPushChunkCompleteInput!) {
  dedicatedModelPushChunkComplete(input: $input) {
    __typename
    ... on DedicatedModelPushChunkCompleteSuccess {
      ok
    }
    ... on UserPermissionError {
      message
    }
  }
}
"""


BaseModelListGql = """
query BaseModelList($dedicatedProjectId: ID!, $conn: BidirectionalConnectionInput) {
  dedicatedProject(id: $dedicatedProjectId) {
    models(conn: $conn) {
      totalCount
      edges {
        node {
          name
          id
          createdAt
        }
      }
    }
  }
}
"""


class ClientTeamMembershipRole(str, Enum):
    OWNER = "OWNER"
    ADMIN = "ADMIN"
    BILLING = "BILLING"
    MEMBER = "MEMBER"


Base64 = NewType("Base64", bytes)


BigInt = NewType("BigInt", str)


URL = NewType("URL", str)


class UserContextResultClientUserTeamsEdgesNode(BaseModel):
    id: str
    name: str | None = None


class UserContextResultClientUserTeamsEdgesProductAccess(BaseModel):
    container: bool | None = None
    dedicated_endpoints: bool | None = Field(alias="dedicatedEndpoints", default=None)
    serverless_endpoints: bool | None = Field(alias="serverlessEndpoints", default=None)


class BidirectionalConnectionInput(BaseModel):
    first: int | None = None
    skip: int
    after: Base64 | None = None
    last: int | None = None
    before: Base64 | None = None


class ClientUserTeamSortsInput(BaseModel):
    ascending: bool
    sort_by: str | None = Field(alias="sortBy", default=None)


class AdapterPushStartResultDedicatedModelPushAdapterStartDedicatedModelPushAdapterStartSuccessAdapter(
    BaseModel
):
    name: str | None = None
    id: str
    created_at: datetime | None = Field(alias="createdAt", default=None)


class AdapterPushStartResultDedicatedModelPushAdapterStartDedicatedModelPushAdapterStartSuccessUploadPlanAdapterConfig(
    BaseModel
):
    required: bool


class AdapterPushStartResultDedicatedModelPushAdapterStartDedicatedModelPushAdapterStartSuccessUploadPlanTokenizerConfig(
    BaseModel
):
    required: bool


class AdapterPushStartResultDedicatedModelPushAdapterStartDedicatedModelPushAdapterStartSuccessUploadPlanSafetensors(
    BaseModel
):
    required: bool


class AdapterPushStartResultDedicatedModelPushAdapterStartUserPermissionError(
    BaseModel
):
    typename__: TypeName = Field(alias="__typename")
    message: str


class FileDescriptorInput(BaseModel):
    digest: str
    filename: str
    size: BigInt


class AdapterPushCompleteResultDedicatedModelPushAdapterCompleteDedicatedModelPushAdapterCompleteSuccessAdapter(
    BaseModel
):
    name: str | None = None
    id: str
    created_at: datetime | None = Field(alias="createdAt", default=None)


class AdapterPushCompleteResultDedicatedModelPushAdapterCompleteUserPermissionError(
    BaseModel
):
    typename__: TypeName = Field(alias="__typename")
    message: str


class FilePushStartResultDedicatedModelPushFileStartDedicatedModelPushFileStartSuccessUploadInfo(
    BaseModel
):
    upload_url: URL = Field(alias="uploadUrl")
    upload_body: dict = Field(alias="uploadBody")


class FilePushStartResultDedicatedModelPushFileStartUserPermissionError(BaseModel):
    typename__: TypeName = Field(alias="__typename")
    message: str


class FilePushStartResultDedicatedModelPushFileStartDedicatedModelPushFileStartAlreadyExistError(
    BaseModel
):
    typename__: TypeName = Field(alias="__typename")
    message: str


class FilePushCompleteResultDedicatedModelPushFileCompleteDedicatedModelPushFileCompleteSuccess(
    BaseModel
):
    typename__: TypeName = Field(alias="__typename")
    ok: bool


class FilePushCompleteResultDedicatedModelPushFileCompleteUserPermissionError(
    BaseModel
):
    typename__: TypeName = Field(alias="__typename")
    message: str


class BasePushStartResultDedicatedModelPushBaseStartDedicatedModelPushBaseStartSuccessModel(
    BaseModel
):
    name: str | None = None
    id: str
    digest: str | None = None
    created_at: datetime | None = Field(alias="createdAt", default=None)


class BasePushStartResultDedicatedModelPushBaseStartDedicatedModelPushBaseStartSuccessUploadPlanConfig(
    BaseModel
):
    required: bool


class BasePushStartResultDedicatedModelPushBaseStartDedicatedModelPushBaseStartSuccessUploadPlanTokenizer(
    BaseModel
):
    required: bool


class BasePushStartResultDedicatedModelPushBaseStartDedicatedModelPushBaseStartSuccessUploadPlanTokenizerConfig(
    BaseModel
):
    required: bool


class BasePushStartResultDedicatedModelPushBaseStartDedicatedModelPushBaseStartSuccessUploadPlanSpecialTokensMap(
    BaseModel
):
    required: bool


class BasePushStartResultDedicatedModelPushBaseStartDedicatedModelPushBaseStartSuccessUploadPlanSafetensors(
    BaseModel
):
    required: bool


class BasePushCompleteResultDedicatedModelPushBaseCompleteDedicatedModelPushBaseCompleteSuccessModel(
    BaseModel
):
    updated_at: datetime | None = Field(alias="updatedAt", default=None)
    name: str | None = None
    id: str
    digest: str | None = None
    created_at: datetime | None = Field(alias="createdAt", default=None)


class ChunkGroupCreateResultDedicatedModelCreateChunkGroupDedicatedModelCreateChunkGroupSuccess(
    BaseModel
):
    chunk_group_id: str = Field(alias="chunkGroupId")


class ChunkGroupCommitResultDedicatedModelCommitChunkGroupDedicatedModelCommitChunkGroupSuccess(
    BaseModel
):
    ok: bool


class ChunkPushStartResultDedicatedModelPushChunkStartDedicatedModelPushChunkStartSuccess(
    BaseModel
):
    typename__: TypeName = Field(alias="__typename")
    upload_url: str = Field(alias="uploadUrl")


class ChunkPushStartResultDedicatedModelPushChunkStartDedicatedModelPushChunkStartAlreadyExistError(
    BaseModel
):
    typename__: TypeName = Field(alias="__typename")
    message: str


class FileChunkInput(BaseModel):
    chunk_group_id: str = Field(alias="chunkGroupId")
    part_number: int = Field(alias="partNumber")
    size: BigInt


class ChunkPushCompleteResultDedicatedModelPushChunkCompleteDedicatedModelPushChunkCompleteSuccess(
    BaseModel
):
    typename__: TypeName = Field(alias="__typename")
    ok: bool


class ChunkPushCompleteResultDedicatedModelPushChunkCompleteUserPermissionError(
    BaseModel
):
    typename__: TypeName = Field(alias="__typename")
    message: str


class FileChunkCompleteInput(BaseModel):
    chunk_group_id: str = Field(alias="chunkGroupId")
    part_number: int = Field(alias="partNumber")
    e_tag: str = Field(alias="eTag")
    size: BigInt


class BaseModelListResultDedicatedProjectModelsEdgesNode(BaseModel):
    name: str | None = None
    id: str
    created_at: datetime | None = Field(alias="createdAt", default=None)


class UserContextResultClientUserTeamsEdges(BaseModel):
    node: UserContextResultClientUserTeamsEdgesNode
    product_access: UserContextResultClientUserTeamsEdgesProductAccess | None = Field(
        alias="productAccess", default=None
    )
    role: ClientTeamMembershipRole | None = None
    joined_at: datetime | None = Field(alias="joinedAt", default=None)
    default: bool | None = None


class BaseModelListVariables(BaseModel):
    dedicated_project_id: str = Field(alias="dedicatedProjectId")
    conn: BidirectionalConnectionInput | None = None


class UserContextVariables(BaseModel):
    conn: BidirectionalConnectionInput | None = None
    sorts: ClientUserTeamSortsInput | None = None


class AdapterPushStartResultDedicatedModelPushAdapterStartDedicatedModelPushAdapterStartSuccessUploadPlan(
    BaseModel
):
    adapter_config: AdapterPushStartResultDedicatedModelPushAdapterStartDedicatedModelPushAdapterStartSuccessUploadPlanAdapterConfig = Field(
        alias="adapterConfig"
    )
    tokenizer_config: (
        AdapterPushStartResultDedicatedModelPushAdapterStartDedicatedModelPushAdapterStartSuccessUploadPlanTokenizerConfig
        | None
    ) = Field(alias="tokenizerConfig", default=None)
    safetensors: list[
        AdapterPushStartResultDedicatedModelPushAdapterStartDedicatedModelPushAdapterStartSuccessUploadPlanSafetensors
    ]


class AdapterModelCreateInput(BaseModel):
    adapter_config: FileDescriptorInput = Field(alias="adapterConfig")
    tokenizer_config: FileDescriptorInput | None = Field(
        alias="tokenizerConfig", default=None
    )
    safetensors: list[FileDescriptorInput]


class DedicatedModelPushFileStartInput(BaseModel):
    model_id: str = Field(alias="modelId")
    file_input: FileDescriptorInput = Field(alias="fileInput")


class DedicatedModelPushFileCompleteInput(BaseModel):
    model_id: str = Field(alias="modelId")
    file_input: FileDescriptorInput = Field(alias="fileInput")


class BaseModelCreateInput(BaseModel):
    config: FileDescriptorInput
    tokenizer: FileDescriptorInput
    tokenizer_config: FileDescriptorInput | None = Field(
        alias="tokenizerConfig", default=None
    )
    special_tokens_map: FileDescriptorInput | None = Field(
        alias="specialTokensMap", default=None
    )
    safetensors: list[FileDescriptorInput]


class DedicatedModelCreateChunkGroupInput(BaseModel):
    model_id: str = Field(alias="modelId")
    file_input: FileDescriptorInput = Field(alias="fileInput")


class DedicatedModelCommitChunkGroupInput(BaseModel):
    model_id: str = Field(alias="modelId")
    chunk_group_id: str = Field(alias="chunkGroupId")
    file_input: FileDescriptorInput = Field(alias="fileInput")


class AdapterPushCompleteResultDedicatedModelPushAdapterCompleteDedicatedModelPushAdapterCompleteSuccess(
    BaseModel
):
    typename__: TypeName = Field(alias="__typename")
    adapter: AdapterPushCompleteResultDedicatedModelPushAdapterCompleteDedicatedModelPushAdapterCompleteSuccessAdapter


class FilePushStartResultDedicatedModelPushFileStartDedicatedModelPushFileStartSuccess(
    BaseModel
):
    typename__: TypeName = Field(alias="__typename")
    upload_info: FilePushStartResultDedicatedModelPushFileStartDedicatedModelPushFileStartSuccessUploadInfo = Field(
        alias="uploadInfo"
    )


FilePushCompleteResultDedicatedModelPushFileComplete = (
    FilePushCompleteResultDedicatedModelPushFileCompleteDedicatedModelPushFileCompleteSuccess
    | FilePushCompleteResultDedicatedModelPushFileCompleteUserPermissionError
)


class BasePushStartResultDedicatedModelPushBaseStartDedicatedModelPushBaseStartSuccessUploadPlan(
    BaseModel
):
    config: BasePushStartResultDedicatedModelPushBaseStartDedicatedModelPushBaseStartSuccessUploadPlanConfig
    tokenizer: BasePushStartResultDedicatedModelPushBaseStartDedicatedModelPushBaseStartSuccessUploadPlanTokenizer
    tokenizer_config: (
        BasePushStartResultDedicatedModelPushBaseStartDedicatedModelPushBaseStartSuccessUploadPlanTokenizerConfig
        | None
    ) = Field(alias="tokenizerConfig", default=None)
    special_tokens_map: (
        BasePushStartResultDedicatedModelPushBaseStartDedicatedModelPushBaseStartSuccessUploadPlanSpecialTokensMap
        | None
    ) = Field(alias="specialTokensMap", default=None)
    safetensors: list[
        BasePushStartResultDedicatedModelPushBaseStartDedicatedModelPushBaseStartSuccessUploadPlanSafetensors
    ]


class BasePushCompleteResultDedicatedModelPushBaseCompleteDedicatedModelPushBaseCompleteSuccess(
    BaseModel
):
    model: BasePushCompleteResultDedicatedModelPushBaseCompleteDedicatedModelPushBaseCompleteSuccessModel


class ChunkGroupCreateResult(BaseModel):
    dedicated_model_create_chunk_group: (
        ChunkGroupCreateResultDedicatedModelCreateChunkGroupDedicatedModelCreateChunkGroupSuccess
    ) = Field(alias="dedicatedModelCreateChunkGroup")


class ChunkGroupCommitResult(BaseModel):
    dedicated_model_commit_chunk_group: (
        ChunkGroupCommitResultDedicatedModelCommitChunkGroupDedicatedModelCommitChunkGroupSuccess
    ) = Field(alias="dedicatedModelCommitChunkGroup")


ChunkPushStartResultDedicatedModelPushChunkStart = (
    ChunkPushStartResultDedicatedModelPushChunkStartDedicatedModelPushChunkStartSuccess
    | ChunkPushStartResultDedicatedModelPushChunkStartDedicatedModelPushChunkStartAlreadyExistError
)


class DedicatedModelPushChunkStartInput(BaseModel):
    model_id: str = Field(alias="modelId")
    file_input: FileChunkInput = Field(alias="fileInput")


ChunkPushCompleteResultDedicatedModelPushChunkComplete = (
    ChunkPushCompleteResultDedicatedModelPushChunkCompleteDedicatedModelPushChunkCompleteSuccess
    | ChunkPushCompleteResultDedicatedModelPushChunkCompleteUserPermissionError
)


class DedicatedModelPushChunkCompleteInput(BaseModel):
    model_id: str = Field(alias="modelId")
    file_input: FileChunkCompleteInput = Field(alias="fileInput")


class BaseModelListResultDedicatedProjectModelsEdges(BaseModel):
    node: BaseModelListResultDedicatedProjectModelsEdgesNode


class UserContextResultClientUserTeams(BaseModel):
    total_count: int = Field(alias="totalCount")
    edges: list[UserContextResultClientUserTeamsEdges]


class AdapterPushStartResultDedicatedModelPushAdapterStartDedicatedModelPushAdapterStartSuccess(
    BaseModel
):
    typename__: TypeName = Field(alias="__typename")
    adapter: AdapterPushStartResultDedicatedModelPushAdapterStartDedicatedModelPushAdapterStartSuccessAdapter
    upload_plan: AdapterPushStartResultDedicatedModelPushAdapterStartDedicatedModelPushAdapterStartSuccessUploadPlan = Field(
        alias="uploadPlan"
    )


class DedicatedModelPushAdapterStartInput(BaseModel):
    project_id: str = Field(alias="projectId")
    name: str | None = None
    base_model_id: str = Field(alias="baseModelId")
    model_structure: AdapterModelCreateInput = Field(alias="modelStructure")


class DedicatedModelPushAdapterCompleteInput(BaseModel):
    adapter_id: str = Field(alias="adapterId")
    model_structure: AdapterModelCreateInput = Field(alias="modelStructure")


class FilePushStartVariables(BaseModel):
    input: DedicatedModelPushFileStartInput


class FilePushCompleteVariables(BaseModel):
    input: DedicatedModelPushFileCompleteInput


class DedicatedModelPushBaseStartInput(BaseModel):
    project_id: str = Field(alias="projectId")
    name: str | None = None
    model_structure: BaseModelCreateInput = Field(alias="modelStructure")


class DedicatedModelPushBaseCompleteInput(BaseModel):
    model_id: str = Field(alias="modelId")
    model_structure: BaseModelCreateInput = Field(alias="modelStructure")


class ChunkGroupCreateVariables(BaseModel):
    input: DedicatedModelCreateChunkGroupInput


class ChunkGroupCommitVariables(BaseModel):
    input: DedicatedModelCommitChunkGroupInput


AdapterPushCompleteResultDedicatedModelPushAdapterComplete = (
    AdapterPushCompleteResultDedicatedModelPushAdapterCompleteUserPermissionError
    | AdapterPushCompleteResultDedicatedModelPushAdapterCompleteDedicatedModelPushAdapterCompleteSuccess
)


FilePushStartResultDedicatedModelPushFileStart = (
    FilePushStartResultDedicatedModelPushFileStartDedicatedModelPushFileStartAlreadyExistError
    | FilePushStartResultDedicatedModelPushFileStartUserPermissionError
    | FilePushStartResultDedicatedModelPushFileStartDedicatedModelPushFileStartSuccess
)


class FilePushCompleteResult(BaseModel):
    dedicated_model_push_file_complete: (
        FilePushCompleteResultDedicatedModelPushFileComplete
    ) = Field(alias="dedicatedModelPushFileComplete")


class BasePushStartResultDedicatedModelPushBaseStartDedicatedModelPushBaseStartSuccess(
    BaseModel
):
    model: BasePushStartResultDedicatedModelPushBaseStartDedicatedModelPushBaseStartSuccessModel
    upload_plan: BasePushStartResultDedicatedModelPushBaseStartDedicatedModelPushBaseStartSuccessUploadPlan = Field(
        alias="uploadPlan"
    )


class BasePushCompleteResult(BaseModel):
    dedicated_model_push_base_complete: (
        BasePushCompleteResultDedicatedModelPushBaseCompleteDedicatedModelPushBaseCompleteSuccess
    ) = Field(alias="dedicatedModelPushBaseComplete")


class ChunkPushStartResult(BaseModel):
    dedicated_model_push_chunk_start: (
        ChunkPushStartResultDedicatedModelPushChunkStart
    ) = Field(alias="dedicatedModelPushChunkStart")


class ChunkPushStartVariables(BaseModel):
    input: DedicatedModelPushChunkStartInput


class ChunkPushCompleteResult(BaseModel):
    dedicated_model_push_chunk_complete: (
        ChunkPushCompleteResultDedicatedModelPushChunkComplete
    ) = Field(alias="dedicatedModelPushChunkComplete")


class ChunkPushCompleteVariables(BaseModel):
    input: DedicatedModelPushChunkCompleteInput


class BaseModelListResultDedicatedProjectModels(BaseModel):
    total_count: int = Field(alias="totalCount")
    edges: list[BaseModelListResultDedicatedProjectModelsEdges]


class UserContextResultClientUser(BaseModel):
    teams: UserContextResultClientUserTeams


AdapterPushStartResultDedicatedModelPushAdapterStart = (
    AdapterPushStartResultDedicatedModelPushAdapterStartDedicatedModelPushAdapterStartSuccess
    | AdapterPushStartResultDedicatedModelPushAdapterStartUserPermissionError
)


class AdapterPushStartVariables(BaseModel):
    input: DedicatedModelPushAdapterStartInput


class AdapterPushCompleteVariables(BaseModel):
    input: DedicatedModelPushAdapterCompleteInput


class BasePushStartVariables(BaseModel):
    input: DedicatedModelPushBaseStartInput


class BasePushCompleteVariables(BaseModel):
    input: DedicatedModelPushBaseCompleteInput


class AdapterPushCompleteResult(BaseModel):
    dedicated_model_push_adapter_complete: (
        AdapterPushCompleteResultDedicatedModelPushAdapterComplete
    ) = Field(alias="dedicatedModelPushAdapterComplete")


class FilePushStartResult(BaseModel):
    dedicated_model_push_file_start: FilePushStartResultDedicatedModelPushFileStart = (
        Field(alias="dedicatedModelPushFileStart")
    )


class BasePushStartResult(BaseModel):
    dedicated_model_push_base_start: BasePushStartResultDedicatedModelPushBaseStartDedicatedModelPushBaseStartSuccess = Field(
        alias="dedicatedModelPushBaseStart"
    )


class BaseModelListResultDedicatedProject(BaseModel):
    models: BaseModelListResultDedicatedProjectModels | None = None


class UserContextResult(BaseModel):
    client_user: UserContextResultClientUser | None = Field(
        alias="clientUser", default=None
    )


class AdapterPushStartResult(BaseModel):
    dedicated_model_push_adapter_start: AdapterPushStartResultDedicatedModelPushAdapterStart = Field(
        alias="dedicatedModelPushAdapterStart"
    )


class BaseModelListResult(BaseModel):
    dedicated_project: BaseModelListResultDedicatedProject | None = Field(
        alias="dedicatedProject", default=None
    )


class BaseSyncStub(AbstractSyncStub):
    def user_context(self, variables: UserContextVariables) -> UserContextResult:
        return self._query(
            UserContextGql, response_model=UserContextResult, variables=variables
        )

    def adapter_push_start(
        self, variables: AdapterPushStartVariables
    ) -> AdapterPushStartResult:
        return self._mutation(
            AdapterPushStartGql,
            response_model=AdapterPushStartResult,
            variables=variables,
        )

    def adapter_push_complete(
        self, variables: AdapterPushCompleteVariables
    ) -> AdapterPushCompleteResult:
        return self._mutation(
            AdapterPushCompleteGql,
            response_model=AdapterPushCompleteResult,
            variables=variables,
        )

    def file_push_start(self, variables: FilePushStartVariables) -> FilePushStartResult:
        return self._mutation(
            FilePushStartGql, response_model=FilePushStartResult, variables=variables
        )

    def file_push_complete(
        self, variables: FilePushCompleteVariables
    ) -> FilePushCompleteResult:
        return self._mutation(
            FilePushCompleteGql,
            response_model=FilePushCompleteResult,
            variables=variables,
        )

    def base_push_start(self, variables: BasePushStartVariables) -> BasePushStartResult:
        return self._mutation(
            BasePushStartGql, response_model=BasePushStartResult, variables=variables
        )

    def base_push_complete(
        self, variables: BasePushCompleteVariables
    ) -> BasePushCompleteResult:
        return self._mutation(
            BasePushCompleteGql,
            response_model=BasePushCompleteResult,
            variables=variables,
        )

    def chunk_group_create(
        self, variables: ChunkGroupCreateVariables
    ) -> ChunkGroupCreateResult:
        return self._mutation(
            ChunkGroupCreateGql,
            response_model=ChunkGroupCreateResult,
            variables=variables,
        )

    def chunk_group_commit(
        self, variables: ChunkGroupCommitVariables
    ) -> ChunkGroupCommitResult:
        return self._mutation(
            ChunkGroupCommitGql,
            response_model=ChunkGroupCommitResult,
            variables=variables,
        )

    def chunk_push_start(
        self, variables: ChunkPushStartVariables
    ) -> ChunkPushStartResult:
        return self._mutation(
            ChunkPushStartGql, response_model=ChunkPushStartResult, variables=variables
        )

    def chunk_push_complete(
        self, variables: ChunkPushCompleteVariables
    ) -> ChunkPushCompleteResult:
        return self._mutation(
            ChunkPushCompleteGql,
            response_model=ChunkPushCompleteResult,
            variables=variables,
        )

    def base_model_list(self, variables: BaseModelListVariables) -> BaseModelListResult:
        return self._query(
            BaseModelListGql, response_model=BaseModelListResult, variables=variables
        )


class BaseAsyncStub(AbstractAsyncStub):
    async def user_context(self, variables: UserContextVariables) -> UserContextResult:
        return await self._query(
            UserContextGql, response_model=UserContextResult, variables=variables
        )

    async def adapter_push_start(
        self, variables: AdapterPushStartVariables
    ) -> AdapterPushStartResult:
        return await self._mutation(
            AdapterPushStartGql,
            response_model=AdapterPushStartResult,
            variables=variables,
        )

    async def adapter_push_complete(
        self, variables: AdapterPushCompleteVariables
    ) -> AdapterPushCompleteResult:
        return await self._mutation(
            AdapterPushCompleteGql,
            response_model=AdapterPushCompleteResult,
            variables=variables,
        )

    async def file_push_start(
        self, variables: FilePushStartVariables
    ) -> FilePushStartResult:
        return await self._mutation(
            FilePushStartGql, response_model=FilePushStartResult, variables=variables
        )

    async def file_push_complete(
        self, variables: FilePushCompleteVariables
    ) -> FilePushCompleteResult:
        return await self._mutation(
            FilePushCompleteGql,
            response_model=FilePushCompleteResult,
            variables=variables,
        )

    async def base_push_start(
        self, variables: BasePushStartVariables
    ) -> BasePushStartResult:
        return await self._mutation(
            BasePushStartGql, response_model=BasePushStartResult, variables=variables
        )

    async def base_push_complete(
        self, variables: BasePushCompleteVariables
    ) -> BasePushCompleteResult:
        return await self._mutation(
            BasePushCompleteGql,
            response_model=BasePushCompleteResult,
            variables=variables,
        )

    async def chunk_group_create(
        self, variables: ChunkGroupCreateVariables
    ) -> ChunkGroupCreateResult:
        return await self._mutation(
            ChunkGroupCreateGql,
            response_model=ChunkGroupCreateResult,
            variables=variables,
        )

    async def chunk_group_commit(
        self, variables: ChunkGroupCommitVariables
    ) -> ChunkGroupCommitResult:
        return await self._mutation(
            ChunkGroupCommitGql,
            response_model=ChunkGroupCommitResult,
            variables=variables,
        )

    async def chunk_push_start(
        self, variables: ChunkPushStartVariables
    ) -> ChunkPushStartResult:
        return await self._mutation(
            ChunkPushStartGql, response_model=ChunkPushStartResult, variables=variables
        )

    async def chunk_push_complete(
        self, variables: ChunkPushCompleteVariables
    ) -> ChunkPushCompleteResult:
        return await self._mutation(
            ChunkPushCompleteGql,
            response_model=ChunkPushCompleteResult,
            variables=variables,
        )

    async def base_model_list(
        self, variables: BaseModelListVariables
    ) -> BaseModelListResult:
        return await self._query(
            BaseModelListGql, response_model=BaseModelListResult, variables=variables
        )
